use bumpalo::Bump;
use crossbeam_utils::CachePadded;

pub struct ScratchBuffer {
    bump: CachePadded<Bump>,
}

impl ScratchBuffer {
    pub fn new() -> Self {
        Self {
            bump: CachePadded::new(Bump::new()),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            bump: CachePadded::new(Bump::with_capacity(capacity)),
        }
    }

    pub fn alloc<T>(&mut self, val: T) -> &mut T {
        self.bump.alloc(val)
    }

    pub fn alloc_slice_clone<T: Clone>(&mut self, src: &[T]) -> &mut [T] {
        self.bump.alloc_slice_clone(src)
    }

    pub fn alloc_slice_copy<T: Copy>(&mut self, src: &[T]) -> &mut [T] {
        self.bump.alloc_slice_copy(src)
    }

    pub fn reset(&mut self) {
        self.bump.reset()
    }
}

impl Default for ScratchBuffer {
    fn default() -> Self {
        Self::new()
    }
}
