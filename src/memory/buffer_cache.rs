use std::{
    collections::HashSet,
    hash::{DefaultHasher, Hash, Hasher},
    iter,
    sync::{Arc, RwLock},
};

pub struct BufferCache<T> {
    shards: Vec<RwLock<Shard<T>>>,
}

impl<T: Hash + Eq> BufferCache<T> {
    const LOG_SHARDS: usize = 6;
    const NUM_SHARDS: usize = 1 << Self::LOG_SHARDS;

    pub fn new() -> Self {
        Self {
            shards: Vec::from_iter(
                iter::repeat_with(|| RwLock::new(Shard::new())).take(Self::NUM_SHARDS),
            ),
        }
    }

    pub fn lookup_or_add(&self, data: Vec<T>) -> Arc<Vec<T>> {
        let mut hasher = DefaultHasher::new();
        let lookup_buffer = Arc::new(data);
        lookup_buffer.hash(&mut hasher);
        let hash = hasher.finish();

        let shard_idx = hash as usize >> (64 - Self::LOG_SHARDS);
        let shard = &self.shards[shard_idx];
        match shard.read().unwrap().get(&lookup_buffer) {
            Some(existing) => existing,
            // This is get or insert because it's possible another thread
            // already inserted right after the above read guard was dropped
            None => shard.write().unwrap().get_or_insert(lookup_buffer),
        }
    }
}

impl<T: Hash + Eq> Default for BufferCache<T> {
    fn default() -> Self {
        Self::new()
    }
}

struct Shard<T> {
    cached_items: HashSet<Arc<Vec<T>>>,
}

impl<T: Hash + Eq> Shard<T> {
    pub fn new() -> Self {
        Self {
            cached_items: HashSet::new(),
        }
    }

    pub fn get(&self, buffer: &Arc<Vec<T>>) -> Option<Arc<Vec<T>>> {
        self.cached_items.get(buffer).map(|buf_arc| buf_arc.clone())
    }

    pub fn get_or_insert(&mut self, buffer: Arc<Vec<T>>) -> Arc<Vec<T>> {
        // The provided Arc should always be dropped...
        match self.get(&buffer) {
            // If the vec already exists in the set, then the provided one is redundant.
            Some(existing) => existing,
            // If it's new, the set will take ownership of the vec from here
            // (Arc is needed for getting it after inserting)
            None => {
                self.cached_items.insert(buffer.clone());
                self.get(&buffer).unwrap()
            }
        }
    }
}
