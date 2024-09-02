use std::{
    hash::{DefaultHasher, Hash, Hasher},
    iter,
    sync::{Arc, LazyLock, RwLock},
};

use bytemuck::Pod;
use hashbrown::HashSet;

use crate::math::{Normal3f, Point2f, Point3f, Vec3f};

pub static USIZE_BUFFER_CACHE: LazyLock<BufferCache<usize>> = LazyLock::new(|| BufferCache::new());
pub static POINT2F_BUFFER_CACHE: LazyLock<BufferCache<Point2f>> =
    LazyLock::new(|| BufferCache::new());
pub static POINT3F_BUFFER_CACHE: LazyLock<BufferCache<Point3f>> =
    LazyLock::new(|| BufferCache::new());
pub static VEC3F_BUFFER_CACHE: LazyLock<BufferCache<Vec3f>> = LazyLock::new(|| BufferCache::new());
pub static NORMAL3F_BUFFER_CACHE: LazyLock<BufferCache<Normal3f>> =
    LazyLock::new(|| BufferCache::new());

pub struct BufferCache<T> {
    shards: Vec<RwLock<Shard<T>>>,
}

impl<T> BufferCache<T> {
    const LOG_SHARDS: usize = 6;
    const NUM_SHARDS: usize = 1 << Self::LOG_SHARDS;

    pub fn new() -> Self {
        Self {
            shards: Vec::from_iter(
                iter::repeat_with(|| RwLock::new(Shard::new())).take(Self::NUM_SHARDS),
            ),
        }
    }
}

impl<T: Pod> BufferCache<T> {
    pub fn lookup_or_add(&self, data: Vec<T>) -> Arc<Vec<T>> {
        let mut hasher = DefaultHasher::new();
        let lookup_buffer = Buffer::new(data);
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

impl<T> Default for BufferCache<T> {
    fn default() -> Self {
        Self::new()
    }
}

struct Shard<T> {
    cached_items: HashSet<Buffer<T>>,
}

impl<T> Shard<T> {
    pub fn new() -> Self {
        Self {
            cached_items: HashSet::new(),
        }
    }
}

impl<T: Pod> Shard<T> {
    pub fn get(&self, buffer: &Buffer<T>) -> Option<Arc<Vec<T>>> {
        self.cached_items.get(buffer).map(|buf| buf.data.clone())
    }

    pub fn get_or_insert(&mut self, buffer: Buffer<T>) -> Arc<Vec<T>> {
        self.cached_items.get_or_insert(buffer).data.clone()
    }
}

struct Buffer<T> {
    data: Arc<Vec<T>>,
}

impl<T> Buffer<T> {
    pub fn new(data: Vec<T>) -> Self {
        Self {
            data: Arc::new(data),
        }
    }
}

impl<T: Pod> PartialEq for Buffer<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.data.len() == other.data.len() {
            let self_bytes: &[u8] = bytemuck::cast_slice(&self.data);
            let other_bytes: &[u8] = bytemuck::cast_slice(&other.data);
            self_bytes == other_bytes
        } else {
            false
        }
    }
}

impl<T: Pod> Eq for Buffer<T> {}

impl<T: Pod> Hash for Buffer<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let bytes: &[u8] = bytemuck::cast_slice(&self.data);
        bytes.hash(state)
    }
}
