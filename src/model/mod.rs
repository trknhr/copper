use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};

use crate::config::ModelSpec;
use crate::loader::Gpt2Weights;
use crate::model::cache::KvCache;
use crate::model::ops::{
    gelu_new, layer_norm, linear_3d, make_causal_mask, make_causal_mask_with_past, softmax_last_dim,
};
use crate::Runtime;

pub mod cache;
pub mod ops;

pub struct Gpt2 {
    spec: ModelSpec,
    w: Gpt2Weights,
    device: Device,
    dtype: DType,
    wte_t: Tensor,
    attn_scale: Tensor,
}

impl Gpt2 {
    pub fn new(spec: ModelSpec, w: Gpt2Weights, rt: &Runtime) -> Result<Self> {
        if rt.dtype != DType::F32 {
            bail!("MVP supports only f32, got {:?}", rt.dtype);
        }
        if !spec.n_embd.is_multiple_of(spec.n_head) {
            bail!(
                "n_embd {} not divisible by n_head {}",
                spec.n_embd,
                spec.n_head
            );
        }
        let wte_t = w.wte_weight.transpose(0, 1)?.contiguous()?;
        let head_dim = spec.n_embd / spec.n_head;
        let attn_scale = Tensor::full((head_dim as f32).powf(-0.5), (), &rt.device)?;

        Ok(Self {
            spec,
            w,
            device: rt.device.clone(),
            dtype: rt.dtype,
            wte_t,
            attn_scale,
        })
    }

    pub fn forward(&self, input_ids: &[u32]) -> Result<Tensor> {
        let seq_len = input_ids.len();
        if seq_len == 0 {
            bail!("input_ids is empty");
        }
        if seq_len > self.spec.n_ctx {
            bail!("seq_len {seq_len} exceeds n_ctx {}", self.spec.n_ctx);
        }

        let ids = Tensor::from_vec(input_ids.to_vec(), (1, seq_len), &self.device)
            .context("create input_ids tensor")?;

        let pos_ids: Vec<u32> = (0..seq_len as u32).collect();
        let pos =
            Tensor::from_vec(pos_ids, (1, seq_len), &self.device).context("create pos_ids")?;

        let tok_emb = self
            .w
            .wte_weight
            .index_select(&ids.flatten_all()?, 0)
            .context("wte index_select")?
            .reshape((1, seq_len, self.spec.n_embd))
            .context("reshape tok_emb")?;

        let pos_emb = self
            .w
            .wpe_weight
            .index_select(&pos.flatten_all()?, 0)
            .context("wpe index_select")?
            .reshape((1, seq_len, self.spec.n_embd))
            .context("reshape pos_emb")?;

        let mut x = tok_emb.add(&pos_emb).context("emb add")?;

        let mask = make_causal_mask(seq_len, &self.device, self.dtype).context("causal mask")?;

        let head_dim = self.spec.n_embd / self.spec.n_head;

        for (i, b) in self.w.blocks.iter().enumerate() {
            let ln1 = layer_norm(
                &x,
                &b.ln_1_weight,
                &b.ln_1_bias,
                self.spec.layer_norm_epsilon,
            )
            .with_context(|| format!("block {i} ln_1"))?;

            let qkv = linear_3d(&ln1, &b.attn_c_attn_weight, &b.attn_c_attn_bias)
                .with_context(|| format!("block {i} attn c_attn"))?;
            let q = qkv.narrow(2, 0, self.spec.n_embd).context("q narrow")?;
            let k = qkv
                .narrow(2, self.spec.n_embd, self.spec.n_embd)
                .context("k narrow")?;
            let v = qkv
                .narrow(2, 2 * self.spec.n_embd, self.spec.n_embd)
                .context("v narrow")?;

            let q = q
                .reshape((1, seq_len, self.spec.n_head, head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
            let k = k
                .reshape((1, seq_len, self.spec.n_head, head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
            let v = v
                .reshape((1, seq_len, self.spec.n_head, head_dim))?
                .transpose(1, 2)?
                .contiguous()?;

            let k_t = k.transpose(2, 3)?.contiguous()?;
            let scores = q
                .matmul(&k_t)
                .with_context(|| format!("block {i} attn matmul"))?
                .broadcast_mul(&self.attn_scale)
                .context("scale")?
                .broadcast_add(&mask)
                .context("mask add")?;

            let probs = softmax_last_dim(&scores).with_context(|| format!("block {i} softmax"))?;
            let ctx = probs
                .matmul(&v)
                .with_context(|| format!("block {i} attn ctx"))?;

            let ctx = ctx
                .transpose(1, 2)?
                .contiguous()?
                .reshape((1, seq_len, self.spec.n_embd))?;

            let attn_out = linear_3d(&ctx, &b.attn_c_proj_weight, &b.attn_c_proj_bias)
                .with_context(|| format!("block {i} attn c_proj"))?;
            x = x
                .add(&attn_out)
                .with_context(|| format!("block {i} attn residual"))?;

            let ln2 = layer_norm(
                &x,
                &b.ln_2_weight,
                &b.ln_2_bias,
                self.spec.layer_norm_epsilon,
            )
            .with_context(|| format!("block {i} ln_2"))?;

            let fc = linear_3d(&ln2, &b.mlp_c_fc_weight, &b.mlp_c_fc_bias)
                .with_context(|| format!("block {i} mlp c_fc"))?;
            let act = gelu_new(&fc).with_context(|| format!("block {i} gelu_new"))?;
            let proj = linear_3d(&act, &b.mlp_c_proj_weight, &b.mlp_c_proj_bias)
                .with_context(|| format!("block {i} mlp c_proj"))?;
            x = x
                .add(&proj)
                .with_context(|| format!("block {i} mlp residual"))?;
        }

        let x = layer_norm(
            &x,
            &self.w.ln_f_weight,
            &self.w.ln_f_bias,
            self.spec.layer_norm_epsilon,
        )
        .context("ln_f")?;

        let logits = x
            .reshape((seq_len, self.spec.n_embd))?
            .matmul(&self.wte_t)?
            .reshape((1, seq_len, self.spec.vocab_size))?;
        Ok(logits)
    }

    pub fn forward_cached(&self, input_ids: &[u32], cache: &mut KvCache) -> Result<Tensor> {
        let cur_len = input_ids.len();
        if cur_len == 0 {
            bail!("input_ids is empty");
        }

        let past_len = cache.past_len;
        let total_len = past_len + cur_len;
        if total_len > self.spec.n_ctx {
            bail!(
                "total_len {} exceeds n_ctx {} (past_len={}, cur_len={})",
                total_len,
                self.spec.n_ctx,
                past_len,
                cur_len
            );
        }
        if cache.layers.len() != self.spec.n_layer {
            bail!(
                "KvCache layer count mismatch: cache has {} layers, model has {}",
                cache.layers.len(),
                self.spec.n_layer
            );
        }
        if cache.n_ctx() != self.spec.n_ctx {
            bail!(
                "KvCache n_ctx mismatch: cache has {}, model has {}",
                cache.n_ctx(),
                self.spec.n_ctx
            );
        }

        let ids = Tensor::from_vec(input_ids.to_vec(), (1, cur_len), &self.device)
            .context("create input_ids tensor")?;

        let pos_ids: Vec<u32> = (past_len as u32..total_len as u32).collect();
        let pos =
            Tensor::from_vec(pos_ids, (1, cur_len), &self.device).context("create pos_ids")?;

        let tok_emb = self
            .w
            .wte_weight
            .index_select(&ids.flatten_all()?, 0)
            .context("wte index_select")?
            .reshape((1, cur_len, self.spec.n_embd))
            .context("reshape tok_emb")?;

        let pos_emb = self
            .w
            .wpe_weight
            .index_select(&pos.flatten_all()?, 0)
            .context("wpe index_select")?
            .reshape((1, cur_len, self.spec.n_embd))
            .context("reshape pos_emb")?;

        let mut x = tok_emb.add(&pos_emb).context("emb add")?;

        let head_dim = self.spec.n_embd / self.spec.n_head;

        let mask = make_causal_mask_with_past(past_len, cur_len, &self.device, self.dtype)
            .context("causal mask with past")?;

        for (i, b) in self.w.blocks.iter().enumerate() {
            let ln1 = layer_norm(
                &x,
                &b.ln_1_weight,
                &b.ln_1_bias,
                self.spec.layer_norm_epsilon,
            )
            .with_context(|| format!("block {i} ln_1"))?;

            let qkv = linear_3d(&ln1, &b.attn_c_attn_weight, &b.attn_c_attn_bias)
                .with_context(|| format!("block {i} attn c_attn"))?;

            let q = qkv.narrow(2, 0, self.spec.n_embd).context("q narrow")?;
            let k = qkv
                .narrow(2, self.spec.n_embd, self.spec.n_embd)
                .context("k narrow")?;
            let v = qkv
                .narrow(2, 2 * self.spec.n_embd, self.spec.n_embd)
                .context("v narrow")?;

            let q = q
                .reshape((1, cur_len, self.spec.n_head, head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
            let k_new = k
                .reshape((1, cur_len, self.spec.n_head, head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
            let v_new = v
                .reshape((1, cur_len, self.spec.n_head, head_dim))?
                .transpose(1, 2)?
                .contiguous()?;

            cache.layers[i]
                .append(&k_new, &v_new, past_len)
                .with_context(|| format!("block {i} append cache"))?;
            let (k_total, v_total) = cache.layers[i]
                .materialize(total_len, &self.device)
                .with_context(|| format!("block {i} materialize cache"))?;

            let k_t = k_total.transpose(2, 3)?.contiguous()?;
            let scores = q
                .matmul(&k_t)
                .with_context(|| format!("block {i} attn matmul"))?
                .broadcast_mul(&self.attn_scale)
                .context("scale")?
                .broadcast_add(&mask)
                .context("mask add")?;

            let probs = softmax_last_dim(&scores).with_context(|| format!("block {i} softmax"))?;
            let ctx = probs
                .matmul(&v_total)
                .with_context(|| format!("block {i} attn ctx"))?;

            let ctx = ctx
                .transpose(1, 2)?
                .contiguous()?
                .reshape((1, cur_len, self.spec.n_embd))?;

            let attn_out = linear_3d(&ctx, &b.attn_c_proj_weight, &b.attn_c_proj_bias)
                .with_context(|| format!("block {i} attn c_proj"))?;
            x = x
                .add(&attn_out)
                .with_context(|| format!("block {i} attn residual"))?;

            let ln2 = layer_norm(
                &x,
                &b.ln_2_weight,
                &b.ln_2_bias,
                self.spec.layer_norm_epsilon,
            )
            .with_context(|| format!("block {i} ln_2"))?;

            let fc = linear_3d(&ln2, &b.mlp_c_fc_weight, &b.mlp_c_fc_bias)
                .with_context(|| format!("block {i} mlp c_fc"))?;
            let act = gelu_new(&fc).with_context(|| format!("block {i} gelu_new"))?;
            let proj = linear_3d(&act, &b.mlp_c_proj_weight, &b.mlp_c_proj_bias)
                .with_context(|| format!("block {i} mlp c_proj"))?;
            x = x
                .add(&proj)
                .with_context(|| format!("block {i} mlp residual"))?;
        }

        let x = layer_norm(
            &x,
            &self.w.ln_f_weight,
            &self.w.ln_f_bias,
            self.spec.layer_norm_epsilon,
        )
        .context("ln_f")?;

        let logits = x
            .reshape((cur_len, self.spec.n_embd))?
            .matmul(&self.wte_t)?
            .reshape((1, cur_len, self.spec.vocab_size))?;

        cache.past_len = total_len;
        Ok(logits)
    }

    pub fn n_layer(&self) -> usize {
        self.spec.n_layer
    }

    pub fn new_kv_cache(&self) -> KvCache {
        let head_dim = self.spec.n_embd / self.spec.n_head;
        KvCache::new(
            self.spec.n_layer,
            self.spec.n_head,
            self.spec.n_ctx,
            head_dim,
        )
    }
}
