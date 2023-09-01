use std::{
    alloc::Layout,
    cell::RefCell,
    cmp::max,
    ptr::{self, NonNull},
};

use crate::L1_CACHE_LINE_SIZE;

use self::buffer::Buffer;

// Similar functionality can be found in crate bumpalo.
#[repr(align(64))]
#[derive(Debug)]
pub struct ScratchBuffer {
    past_buffers: RefCell<Vec<Buffer>>,
    current_buffer: RefCell<Buffer>,
}

impl ScratchBuffer {
    pub const ALIGN: usize = L1_CACHE_LINE_SIZE;

    pub fn new() -> Self {
        Self::with_size(256).unwrap()
    }

    pub fn with_size(size: usize) -> Option<Self> {
        let first_sub_buffer = Buffer::new(size)?;

        Some(Self {
            past_buffers: RefCell::new(Vec::new()),
            current_buffer: RefCell::new(first_sub_buffer),
        })
    }

    #[must_use]
    pub fn alloc_layout(&self, layout: Layout) -> NonNull<u8> {
        // Borrow buffer list for mutation
        // Try to allocate from current buffer
        let allow_ret = self.current_buffer.borrow_mut().try_alloc(layout);
        // Borrow has ended here; following borrows will be okay

        if let Some(ptr) = allow_ret {
            // Alloc succeeded, return it
            ptr
        } else {
            // Alloc failed, realloc buffer and try again
            self.realloc(layout.size());
            self.current_buffer.borrow_mut().try_alloc(layout).expect(
                "Newly added buffer should've been one with enough space for this allocation",
            )
        }
    }

    #[must_use]
    pub fn alloc_object<T>(&self, t: T) -> &mut T {
        self.alloc_with(|| t)
    }

    #[must_use]
    pub fn alloc_with<F, T>(&self, f: F) -> &mut T
    where
        F: FnOnce() -> T,
    {
        #[inline(always)]
        unsafe fn inner_writer<T, F>(ptr: *mut T, f: F)
        where
            F: FnOnce() -> T,
        {
            // From crate bumpalo:
            //
            // This function is translated as:
            // - allocate space for a T on the stack
            // - call f() with the return value being put onto this stack space
            // - memcpy from the stack to the heap
            //
            // Ideally we want LLVM to always realize that doing a stack
            // allocation is unnecessary and optimize the code so it writes
            // directly into the heap instead. It seems we get it to realize
            // this most consistently if we put this critical line into it's
            // own function instead of inlining it into the surrounding code.
            ptr::write(ptr, f())
        }

        // UNSAFE: Direct write to pointer
        unsafe {
            let ptr = self.alloc_layout(Layout::new::<T>());
            let t_ptr = ptr.as_ptr() as *mut T;
            inner_writer(t_ptr, f);

            &mut *t_ptr
        }
    }

    /// Drop all of `self`'s buffer space, invalidating any memory obtained from `self`.
    pub fn reset(&mut self) {
        let last_size = self.current_buffer.borrow().size();
        // Drop all the buffers, past and current, and therefore their owned memory.
        // Then make a new current buffer with the last used size.
        self.past_buffers.get_mut().clear();
        self.current_buffer.replace(Buffer::new(last_size).unwrap());
    }

    fn realloc(&self, min_size: usize) {
        let new_alloc_size = max(2 * min_size, self.current_buffer.borrow().size() + min_size);
        let new_sub_buffer = Buffer::new(new_alloc_size).unwrap();

        let prev_buffer = self.current_buffer.replace(new_sub_buffer);
        self.past_buffers.borrow_mut().push(prev_buffer);
    }
}

impl Default for ScratchBuffer {
    fn default() -> Self {
        Self::new()
    }
}

mod buffer {
    use std::{
        alloc::{self, Layout},
        ptr::NonNull,
    };

    use crate::L1_CACHE_LINE_SIZE;

    #[repr(C, align(64))]
    #[derive(Debug)]
    pub struct Buffer {
        ptr: NonNull<u8>,
        layout: Layout,
        offset: usize,
    }

    impl Buffer {
        const ALIGN: usize = L1_CACHE_LINE_SIZE;

        pub fn new(size: usize) -> Option<Self> {
            let layout = Layout::from_size_align(size, Self::ALIGN).ok()?;

            // UNSAFE: Manual allocation and NonNull
            unsafe {
                let ptr = alloc::alloc(layout);
                if ptr.is_null() {
                    alloc::handle_alloc_error(layout);
                }

                Some(Self {
                    ptr: NonNull::new_unchecked(ptr),
                    offset: 0,
                    layout,
                })
            }
        }

        pub fn try_alloc(&mut self, layout: Layout) -> Option<NonNull<u8>> {
            let (size, align) = (layout.size(), layout.align());

            // Will be the offset in bytes of the returned block
            // (the starting byte)
            let ret_offset = if self.offset % align != 0 {
                // If unaligned, align
                self.offset + align + (self.offset % align)
            } else {
                self.offset
            };

            // If there's enough space in the buffer, return the ptr.
            // Otherwise, nothing to return.
            if ret_offset + size <= self.layout.size() {
                self.offset = ret_offset;

                // UNSAFE: Pointer arithmetic
                unsafe {
                    let ret_ptr = self.ptr.as_ptr().add(self.offset);
                    self.offset = ret_offset + size;
                    Some(NonNull::new_unchecked(ret_ptr))
                }
            } else {
                None
            }
        }

        #[inline]
        pub fn size(&self) -> usize {
            self.layout.size()
        }

        #[cfg(test)]
        pub fn offset(&self) -> usize {
            self.offset
        }
    }

    impl Drop for Buffer {
        fn drop(&mut self) {
            // UNSAFE: Manual deallocation
            unsafe {
                alloc::dealloc(self.ptr.as_ptr(), self.layout);
            }
        }
    }
}

#[cfg(test)]
mod test {
    use crate::math::{matrix4x4::Matrix4x4, vec3::Vec3i};

    use super::*;

    #[test]
    fn alloc_with_then_reset() {
        let mut buffer = ScratchBuffer::new();
        let f = || Matrix4x4::IDENTITY;
        for _ in 0..1000 {
            let returned_ref = buffer.alloc_with(f);
            assert_eq!(*returned_ref, Matrix4x4::IDENTITY);
        }

        buffer.reset();

        assert!(buffer.past_buffers.borrow().is_empty());
        assert_eq!(buffer.current_buffer.borrow().offset(), 0);
    }

    #[test]
    fn alloc_mixed_size() {
        let mut buffer = ScratchBuffer::new();
        let f_mat = || Matrix4x4::IDENTITY;
        let f_vec = || Vec3i::new(1, 2, 3);
        for _ in 0..1000 {
            let returned_mat = buffer.alloc_with(f_mat);
            assert_eq!(*returned_mat, Matrix4x4::IDENTITY);
            let returned_vec = buffer.alloc_with(f_vec);
            assert_eq!(*returned_vec, Vec3i::new(1, 2, 3));
        }

        buffer.reset();
    }

    #[test]
    fn alloc_mixed_size_with_reuse() {
        let mut buffer = ScratchBuffer::new();
        let f_mat = || Matrix4x4::IDENTITY;
        let f_vec = || Vec3i::new(1, 2, 3);
        for _ in 0..1000 {
            let returned_mat = buffer.alloc_with(f_mat);
            assert_eq!(*returned_mat, Matrix4x4::IDENTITY);
            let returned_vec = buffer.alloc_with(f_vec);
            assert_eq!(*returned_vec, Vec3i::new(1, 2, 3));
        }

        buffer.reset();

        for _ in 0..1000 {
            let returned_vec = buffer.alloc_with(f_vec);
            assert_eq!(*returned_vec, Vec3i::new(1, 2, 3));
            let returned_mat = buffer.alloc_with(f_mat);
            assert_eq!(*returned_mat, Matrix4x4::IDENTITY);
        }
    }
}

#[cfg(test)]
mod bench {
    use bumpalo::Bump;

    use super::*;

    use std::{hint::black_box, time::Instant};

    use crate::{
        math::square_matrix::SquareMatrix,
        util::spectrum::{ConstantSpectrum, DenselySampledSpectrum},
    };

    #[test]
    fn bench_alloc_mixed_size_100000x2_each() {
        println!("[BENCH] bench_alloc_mixed_size_100000x2_each");

        let f_sqr_mat = || black_box(SquareMatrix::<3>::default());
        let f_spectrum = || {
            black_box(DenselySampledSpectrum::new(
                &ConstantSpectrum::new(5.0),
                None,
                None,
            ))
        };

        let start = Instant::now(); // START

        let mut buffer = ScratchBuffer::new();

        for _ in 0..100000 {
            let spec = black_box(buffer.alloc_with(f_spectrum));
            let mat = black_box(buffer.alloc_with(f_sqr_mat));
            black_box(spec);
            black_box(mat);
        }

        // buffer.reset();

        for _ in 0..100000 {
            let mat = black_box(buffer.alloc_with(f_sqr_mat));
            let spec = black_box(buffer.alloc_with(f_spectrum));
            black_box(mat);
            black_box(spec);
        }

        buffer.reset();

        let end = Instant::now(); // END
        println!(
            "[BENCH] ScratchBuffer took {:?}ms",
            (end - start).as_millis()
        );

        // BASELINE - stack
        let start = Instant::now(); // START

        for _ in 0..100000 {
            let spec = black_box(buffer.alloc_with(f_spectrum));
            let mat = black_box(buffer.alloc_with(f_sqr_mat));
            black_box(spec);
            black_box(mat);
        }

        for _ in 0..100000 {
            let mat = black_box(buffer.alloc_with(f_sqr_mat));
            let spec = black_box(buffer.alloc_with(f_spectrum));
            black_box(mat);
            black_box(spec);
        }

        let end = Instant::now();
        println!(
            "[BENCH] Default behavior took {:?}ms",
            (end - start).as_millis()
        );

        // BASELINE - bumpalo
        let start = Instant::now(); // START

        let bump = Bump::new();

        for _ in 0..100000 {
            let spec = black_box(bump.alloc_with(f_spectrum));
            let mat = black_box(bump.alloc_with(f_sqr_mat));
            black_box(spec);
            black_box(mat);
        }

        for _ in 0..100000 {
            let mat = black_box(bump.alloc_with(f_sqr_mat));
            let spec = black_box(bump.alloc_with(f_spectrum));
            black_box(mat);
            black_box(spec);
        }

        let end = Instant::now();
        println!("[BENCH] Bumpalo took {:?}ms", (end - start).as_millis());
    }
}
