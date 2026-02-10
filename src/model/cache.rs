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
        for layer in &mut self.layers {
            *layer = None;
        }
        self.past_len = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clear_resets_past_len_only() {
        let mut cache = KvCache::new(2);
        cache.past_len = 5;
        cache.clear();
        assert_eq!(cache.past_len, 0);
        assert_eq!(cache.layers.len(), 2);
        assert!(cache.layers.iter().all(|x| x.is_none()));
    }
}
