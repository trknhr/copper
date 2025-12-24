use candle_core::Tensor;

#[derive(Debug, Clone)]
pub struct LayerKv {
    pub k: Tensor,
    pub v: Tensor,
}

#[derive(Debug, Clone)]
pub struct KvCache {
    pub layers: Vec<Option<LayerKv>>,
    pub past_len: usize,
}

impl KvCache {
    pub fn new(n_layer: usize) -> Self {
        Self {
            layers: vec![None; n_layer],
            past_len: 0,
        }
    }

    pub fn clear(&mut self) {
        for layer in self.layers.iter_mut() {
            *layer = None;
        }
        self.past_len = 0;
    }
}
