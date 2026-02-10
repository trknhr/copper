use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use memmap2::Mmap;
use safetensors::{tensor::TensorView, Dtype, SafeTensors};

use crate::{config::ModelSpec, Runtime};

#[derive(Debug, Clone, Copy)]
enum KeyStyle {
    FullTransformer,
    NoTransformer,
}

#[derive(Debug, Clone)]
struct KeyResolver {
    prefix: String,
    style: KeyStyle,
}

impl KeyResolver {
    fn infer(st: &SafeTensors<'_>) -> Result<Self> {
        if st.tensor("transformer.wte.weight").is_ok() {
            return Ok(Self {
                prefix: String::new(),
                style: KeyStyle::FullTransformer,
            });
        }
        let names = st.names();
        if let Some(name) = names
            .iter()
            .find(|n| n.as_str().ends_with("transformer.wte.weight"))
        {
            let prefix = name
                .as_str()
                .trim_end_matches("transformer.wte.weight")
                .to_string();
            return Ok(Self {
                prefix,
                style: KeyStyle::FullTransformer,
            });
        }

        if st.tensor("wte.weight").is_ok() {
            return Ok(Self {
                prefix: String::new(),
                style: KeyStyle::NoTransformer,
            });
        }
        if let Some(name) = names.iter().find(|n| n.as_str().ends_with("wte.weight")) {
            let prefix = name.as_str().trim_end_matches("wte.weight").to_string();
            return Ok(Self {
                prefix,
                style: KeyStyle::NoTransformer,
            });
        }

        let mut sample = st
            .names()
            .into_iter()
            .take(50)
            .map(|s| s.as_str().to_string())
            .collect::<Vec<_>>();
        sample.sort();
        bail!(
            "could not infer safetensors key style (expected to find a key ending with transformer.wte.weight or wte.weight). Sample keys: {sample:?}"
        );
    }

    fn resolve(&self, base: &str) -> Result<String> {
        let tail = match self.style {
            KeyStyle::FullTransformer => base.to_string(),
            KeyStyle::NoTransformer => base
                .strip_prefix("transformer.")
                .ok_or_else(|| anyhow::anyhow!("expected transformer.* base key, got {base}"))?
                .to_string(),
        };
        Ok(format!("{}{}", self.prefix, tail))
    }
}

#[derive(Debug, Clone)]
pub struct BlockWeights {
    pub ln_1_weight: Tensor,
    pub ln_1_bias: Tensor,
    pub attn_c_attn_weight: Tensor,
    pub attn_c_attn_bias: Tensor,
    pub attn_c_proj_weight: Tensor,
    pub attn_c_proj_bias: Tensor,
    pub ln_2_weight: Tensor,
    pub ln_2_bias: Tensor,
    pub mlp_c_fc_weight: Tensor,
    pub mlp_c_fc_bias: Tensor,
    pub mlp_c_proj_weight: Tensor,
    pub mlp_c_proj_bias: Tensor,
}

#[derive(Debug, Clone)]
pub struct Gpt2Weights {
    pub wte_weight: Tensor,
    pub wpe_weight: Tensor,
    pub blocks: Vec<BlockWeights>,
    pub ln_f_weight: Tensor,
    pub ln_f_bias: Tensor,
}

pub fn load_gpt2_weights(model_dir: &Path, rt: &Runtime) -> Result<Gpt2Weights> {
    if rt.dtype != DType::F32 {
        bail!("MVP supports only f32, got {:?}", rt.dtype);
    }

    let spec = crate::config::load_model_spec(model_dir).context("load config.json for shapes")?;
    let file = find_single_safetensors(model_dir)?;
    let f = std::fs::File::open(&file).with_context(|| format!("open {file:?}"))?;
    let mmap = unsafe { Mmap::map(&f).with_context(|| format!("mmap {file:?}"))? };
    let st = SafeTensors::deserialize(&mmap).context("deserialize safetensors")?;
    let ks = KeyResolver::infer(&st).context("infer safetensors key style")?;

    let wte_weight = get_f32(
        &st,
        &ks,
        "transformer.wte.weight",
        &[spec.vocab_size, spec.n_embd],
        &rt.device,
    )?;
    let wpe_weight = get_f32(
        &st,
        &ks,
        "transformer.wpe.weight",
        &[spec.n_positions(), spec.n_embd],
        &rt.device,
    )?;

    let mut blocks = Vec::with_capacity(spec.n_layer);
    for i in 0..spec.n_layer {
        let p = format!("transformer.h.{i}.");
        let ln_1_weight = get_f32(
            &st,
            &ks,
            &(p.clone() + "ln_1.weight"),
            &[spec.n_embd],
            &rt.device,
        )?;
        let ln_1_bias = get_f32(
            &st,
            &ks,
            &(p.clone() + "ln_1.bias"),
            &[spec.n_embd],
            &rt.device,
        )?;
        let attn_c_attn_weight = get_f32(
            &st,
            &ks,
            &(p.clone() + "attn.c_attn.weight"),
            &[spec.n_embd, 3 * spec.n_embd],
            &rt.device,
        )?;
        let attn_c_attn_bias = get_f32(
            &st,
            &ks,
            &(p.clone() + "attn.c_attn.bias"),
            &[3 * spec.n_embd],
            &rt.device,
        )?;
        let attn_c_proj_weight = get_f32(
            &st,
            &ks,
            &(p.clone() + "attn.c_proj.weight"),
            &[spec.n_embd, spec.n_embd],
            &rt.device,
        )?;
        let attn_c_proj_bias = get_f32(
            &st,
            &ks,
            &(p.clone() + "attn.c_proj.bias"),
            &[spec.n_embd],
            &rt.device,
        )?;
        let ln_2_weight = get_f32(
            &st,
            &ks,
            &(p.clone() + "ln_2.weight"),
            &[spec.n_embd],
            &rt.device,
        )?;
        let ln_2_bias = get_f32(
            &st,
            &ks,
            &(p.clone() + "ln_2.bias"),
            &[spec.n_embd],
            &rt.device,
        )?;
        let inner = spec.n_inner();
        let mlp_c_fc_weight = get_f32(
            &st,
            &ks,
            &(p.clone() + "mlp.c_fc.weight"),
            &[spec.n_embd, inner],
            &rt.device,
        )?;
        let mlp_c_fc_bias = get_f32(
            &st,
            &ks,
            &(p.clone() + "mlp.c_fc.bias"),
            &[inner],
            &rt.device,
        )?;
        let mlp_c_proj_weight = get_f32(
            &st,
            &ks,
            &(p.clone() + "mlp.c_proj.weight"),
            &[inner, spec.n_embd],
            &rt.device,
        )?;
        let mlp_c_proj_bias = get_f32(
            &st,
            &ks,
            &(p + "mlp.c_proj.bias"),
            &[spec.n_embd],
            &rt.device,
        )?;

        blocks.push(BlockWeights {
            ln_1_weight,
            ln_1_bias,
            attn_c_attn_weight,
            attn_c_attn_bias,
            attn_c_proj_weight,
            attn_c_proj_bias,
            ln_2_weight,
            ln_2_bias,
            mlp_c_fc_weight,
            mlp_c_fc_bias,
            mlp_c_proj_weight,
            mlp_c_proj_bias,
        });
    }

    let ln_f_weight = get_f32(
        &st,
        &ks,
        "transformer.ln_f.weight",
        &[spec.n_embd],
        &rt.device,
    )?;
    let ln_f_bias = get_f32(
        &st,
        &ks,
        "transformer.ln_f.bias",
        &[spec.n_embd],
        &rt.device,
    )?;

    Ok(Gpt2Weights {
        wte_weight,
        wpe_weight,
        blocks,
        ln_f_weight,
        ln_f_bias,
    })
}

fn find_single_safetensors(model_dir: &Path) -> Result<PathBuf> {
    let candidate = model_dir.join("model.safetensors");
    if candidate.exists() {
        return Ok(candidate);
    }
    let mut found = vec![];
    for entry in std::fs::read_dir(model_dir).with_context(|| format!("read_dir {model_dir:?}"))? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("safetensors") {
            found.push(path);
        }
    }
    match found.len() {
        1 => Ok(found.remove(0)),
        0 => bail!("no .safetensors found under {model_dir:?} (expected model.safetensors)"),
        _ => bail!(
            "multiple .safetensors found under {model_dir:?}, MVP supports single-file models"
        ),
    }
}

fn get_f32(
    st: &SafeTensors<'_>,
    ks: &KeyResolver,
    base: &str,
    shape: &[usize],
    device: &Device,
) -> Result<Tensor> {
    let name = ks.resolve(base).context("resolve tensor key")?;
    let t = st.tensor(&name).with_context(|| {
        let mut sample = st
            .names()
            .into_iter()
            .take(50)
            .map(|s| s.as_str().to_string())
            .collect::<Vec<_>>();
        sample.sort();
        format!("missing tensor {name} (from base {base}). Sample keys: {sample:?}")
    })?;
    tensor_view_to_candle_f32(&t, &name, shape, device)
}

fn tensor_view_to_candle_f32(
    t: &TensorView<'_>,
    name: &str,
    expected: &[usize],
    device: &Device,
) -> Result<Tensor> {
    if t.dtype() != Dtype::F32 {
        bail!("tensor {name} has dtype {:?}, expected f32", t.dtype());
    }
    let actual = t.shape();
    if actual != expected {
        bail!("tensor {name} has shape {actual:?}, expected {expected:?}");
    }
    let bytes = t.data();
    if !bytes.len().is_multiple_of(4) {
        bail!(
            "tensor {name} has invalid byte length {} for f32 data",
            bytes.len()
        );
    }
    let mut data = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Tensor::from_vec(data, expected, device).with_context(|| format!("create candle tensor {name}"))
}

pub fn load_model_spec(model_dir: &Path) -> Result<ModelSpec> {
    crate::config::load_model_spec(model_dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn missing_safetensors_is_error() {
        let dir = tempdir().expect("tempdir");
        let err = find_single_safetensors(dir.path()).expect_err("should fail");
        let msg = format!("{err:#}");
        assert!(msg.contains("no .safetensors found"));
    }

    #[test]
    fn multiple_safetensors_is_error() {
        let dir = tempdir().expect("tempdir");
        std::fs::write(dir.path().join("a.safetensors"), b"").expect("write a");
        std::fs::write(dir.path().join("b.safetensors"), b"").expect("write b");
        let err = find_single_safetensors(dir.path()).expect_err("should fail");
        let msg = format!("{err:#}");
        assert!(msg.contains("multiple .safetensors found"));
    }
}
