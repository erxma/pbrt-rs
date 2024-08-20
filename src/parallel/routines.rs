use std::cmp::Ordering;

pub trait ParallelSliceMut<T: Send> {
    fn par_sort_by<F>(&mut self, compare: F)
    where
        F: Fn(&T, &T) -> Ordering + Sync,
    {
    }
}

impl<T: Send> ParallelSliceMut<T> for [T] {}
