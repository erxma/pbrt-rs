use std::ops::{Index, IndexMut};

use crate::geometry::bounds::Bounds2i;

use super::point::Point2i;

#[derive(Clone, Debug)]
pub struct Array2D<T> {
    extent: Bounds2i,
    values: Vec<T>,
}

impl<T> Array2D<T> {
    pub fn fill_default(extent: Bounds2i) -> Self
    where
        T: Default,
    {
        todo!()
    }
}

impl<T> Index<Point2i> for Array2D<T> {
    type Output = T;

    fn index(&self, index: Point2i) -> &Self::Output {
        todo!()
    }
}

impl<T> IndexMut<Point2i> for Array2D<T> {
    fn index_mut(&mut self, index: Point2i) -> &mut Self::Output {
        todo!()
    }
}
