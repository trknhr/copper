use anyhow::{bail, Result};
use candle_core::{Device, Tensor};

#[derive(Debug, Clone)]
pub struct LayerKv {
    n_head: usize,
    n_ctx: usize,
    head_dim: usize,
    k_data: Vec<f32>,
    v_data: Vec<f32>,
}

impl LayerKv {
    pub fn new(n_head: usize, n_ctx: usize, head_dim: usize) -> Self {
        let cap = n_head * n_ctx * head_dim;
        Self {
            n_head,
            n_ctx,
            head_dim,
            k_data: vec![0.0; cap],
            v_data: vec![0.0; cap],
        }
    }

    pub fn append(&mut self, k_new: &Tensor, v_new: &Tensor, past_len: usize) -> Result<()> {
        let cur_len = k_new.dims()[2];
        if cur_len == 0 {
            bail!("k/v append requires cur_len > 0");
        }
        if past_len + cur_len > self.n_ctx {
            bail!(
                "k/v append exceeds n_ctx: past_len {} + cur_len {} > {}",
                past_len,
                cur_len,
                self.n_ctx
            );
        }

        let k_flat: Vec<f32> = k_new.flatten_all()?.to_vec1()?;
        let v_flat: Vec<f32> = v_new.flatten_all()?.to_vec1()?;
        let src_stride = cur_len * self.head_dim;
        let dst_stride = self.n_ctx * self.head_dim;
        for h in 0..self.n_head {
            let src_base = h * src_stride;
            let dst_base = h * dst_stride + past_len * self.head_dim;
            for t in 0..cur_len {
                let src = src_base + t * self.head_dim;
                let dst = dst_base + t * self.head_dim;
                self.k_data[dst..dst + self.head_dim]
                    .copy_from_slice(&k_flat[src..src + self.head_dim]);
                self.v_data[dst..dst + self.head_dim]
                    .copy_from_slice(&v_flat[src..src + self.head_dim]);
            }
        }
        Ok(())
    }

    pub fn materialize(&self, total_len: usize, device: &Device) -> Result<(Tensor, Tensor)> {
        if total_len == 0 || total_len > self.n_ctx {
            bail!("invalid total_len {} for n_ctx {}", total_len, self.n_ctx);
        }
        let mut k_out = vec![0.0; self.n_head * total_len * self.head_dim];
        let mut v_out = vec![0.0; self.n_head * total_len * self.head_dim];
        let src_stride = self.n_ctx * self.head_dim;
        let dst_stride = total_len * self.head_dim;
        for h in 0..self.n_head {
            let src = h * src_stride;
            let dst = h * dst_stride;
            let len = total_len * self.head_dim;
            k_out[dst..dst + len].copy_from_slice(&self.k_data[src..src + len]);
            v_out[dst..dst + len].copy_from_slice(&self.v_data[src..src + len]);
        }
        let k = Tensor::from_vec(k_out, (1, self.n_head, total_len, self.head_dim), device)?;
        let v = Tensor::from_vec(v_out, (1, self.n_head, total_len, self.head_dim), device)?;
        Ok((k, v))
    }
}

#[derive(Debug, Clone)]
pub struct KvCache {
    pub layers: Vec<LayerKv>,
    pub past_len: usize,
    n_ctx: usize,
}

impl KvCache {
    pub fn new(n_layer: usize, n_head: usize, n_ctx: usize, head_dim: usize) -> Self {
        let mut layers = Vec::with_capacity(n_layer);
        for _ in 0..n_layer {
            layers.push(LayerKv::new(n_head, n_ctx, head_dim));
        }
        Self {
            layers,
            past_len: 0,
            n_ctx,
        }
    }

    pub fn clear(&mut self) {
        self.past_len = 0;
    }

    pub fn n_ctx(&self) -> usize {
        self.n_ctx
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn clear_resets_past_len_only() {
        let mut cache = KvCache::new(2, 3, 8, 4);
        cache.past_len = 5;
        cache.clear();
        assert_eq!(cache.past_len, 0);
        assert_eq!(cache.layers.len(), 2);
    }

    #[test]
    fn append_and_materialize_shapes() -> Result<()> {
        let device = Device::Cpu;
        let mut layer = LayerKv::new(2, 8, 4);
        let k = Tensor::from_vec(vec![1.0_f32; 2 * 3 * 4], (1, 2, 3, 4), &device)?;
        let v = Tensor::from_vec(vec![2.0_f32; 2 * 3 * 4], (1, 2, 3, 4), &device)?;
        layer.append(&k, &v, 0)?;
        let (k_total, v_total) = layer.materialize(3, &device)?;
        assert_eq!(k_total.dims(), &[1, 2, 3, 4]);
        assert_eq!(v_total.dims(), &[1, 2, 3, 4]);
        Ok(())
    }
}
