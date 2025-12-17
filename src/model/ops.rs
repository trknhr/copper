use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::ops::softmax;

pub fn linear_3d(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let shape = x.dims();
    if shape.len() != 3 {
        bail!("linear_3d expects rank-3 input, got {:?}", shape);
    }
    let (b, s, in_dim) = (shape[0], shape[1], shape[2]);
    let wshape = weight.dims();
    if wshape.len() != 2 {
        bail!("linear_3d expects rank-2 weight, got {:?}", wshape);
    }
    let (w_in, w_out) = (wshape[0], wshape[1]);
    if w_in != in_dim {
        bail!("linear_3d weight in_dim mismatch: x {in_dim} vs w {w_in}");
    }
    let bshape = bias.dims();
    if bshape != [w_out] {
        bail!(
            "linear_3d bias shape mismatch: expected [{w_out}] got {:?}",
            bshape
        );
    }
    let y = x
        .reshape((b * s, in_dim))?
        .matmul(weight)?
        .broadcast_add(bias)?;
    y.reshape((b, s, w_out)).map_err(Into::into)
}

pub fn layer_norm(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f64) -> Result<Tensor> {
    let dims = x.dims();
    if dims.is_empty() {
        bail!("layer_norm expects non-scalar tensor");
    }
    let last = *dims.last().unwrap();
    let wshape = weight.dims();
    let bshape = bias.dims();
    if wshape != [last] || bshape != [last] {
        bail!(
            "layer_norm weight/bias shape mismatch: expected [{last}], got w={wshape:?} b={bshape:?}"
        );
    }

    let mean = x.mean_keepdim(candle_core::D::Minus1)?;
    let centered = x.broadcast_sub(&mean)?;
    let var = centered.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
    let eps = Tensor::full(eps as f32, (), var.device())?;
    let denom = var.broadcast_add(&eps)?.sqrt()?;
    let normed = centered.broadcast_div(&denom)?;
    normed
        .broadcast_mul(weight)?
        .broadcast_add(bias)
        .map_err(Into::into)
}

pub fn gelu_new(x: &Tensor) -> Result<Tensor> {
    let x3 = x.mul(x)?.mul(x)?;
    let a = Tensor::full(0.044715_f32, (), x.device())?;
    let inner = x.add(&x3.broadcast_mul(&a)?)?;
    let c = Tensor::full(0.79788456_f32, (), x.device())?; // sqrt(2/pi)
    let tanh = inner.broadcast_mul(&c)?.tanh()?;
    let one = Tensor::full(1f32, (), x.device())?;
    let half = Tensor::full(0.5_f32, (), x.device())?;
    let one_plus = tanh.broadcast_add(&one)?;
    x.mul(&one_plus)?.broadcast_mul(&half).map_err(Into::into)
}

pub fn softmax_last_dim(x: &Tensor) -> Result<Tensor> {
    softmax(x, candle_core::D::Minus1).map_err(Into::into)
}

pub fn make_causal_mask(seq_len: usize, device: &Device, dtype: DType) -> Result<Tensor> {
    if dtype != DType::F32 {
        bail!("make_causal_mask only supports f32 dtype, got {:?}", dtype);
    }
    let mut data = vec![0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            data[i * seq_len + j] = -1e4;
        }
    }
    Tensor::from_vec(data, (1, 1, seq_len, seq_len), device).map_err(Into::into)
}
