use std::{
    ops::{Index, IndexMut},
    vec,
};

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
        let n = extent.area() as usize;
        let mut values = Vec::new();
        values.resize_with(n, Default::default);
        Self { extent, values }
    }

    pub fn fill_value(extent: Bounds2i, value: T) -> Self
    where
        T: Clone,
    {
        let n = extent.area() as usize;
        let values = vec![value; n];
        Self { extent, values }
    }

    pub fn num_values(&self) -> usize {
        self.extent.area() as usize
    }

    pub fn x_size(&self) -> usize {
        (self.extent.p_max.x() - self.extent.p_min.x()) as usize
    }

    pub fn y_size(&self) -> usize {
        (self.extent.p_max.y() - self.extent.p_min.y()) as usize
    }
}

impl<T> Index<Point2i> for Array2D<T> {
    type Output = T;

    fn index(&self, p: Point2i) -> &Self::Output {
        debug_assert!(p.inside_exclusive(self.extent));

        let x = (p.x() - self.extent.p_min.x()) as usize;
        let y = (p.y() - self.extent.p_min.y()) as usize;

        &self.values[y * self.x_size() + x]
    }
}

impl<T> IndexMut<Point2i> for Array2D<T> {
    fn index_mut(&mut self, p: Point2i) -> &mut Self::Output {
        debug_assert!(p.inside_exclusive(self.extent));

        let x = (p.x() - self.extent.p_min.x()) as usize;
        let y = (p.y() - self.extent.p_min.y()) as usize;
        let idx = y * self.x_size() + x;

        &mut self.values[idx]
    }
}

// Iterator is same as the value vec's,
// equivalent to going through each row, "left to right" within each row
impl<T> IntoIterator for Array2D<T> {
    type Item = T;
    type IntoIter = Array2DIterator<T>;

    fn into_iter(self) -> Self::IntoIter {
        Array2DIterator(self.values.into_iter())
    }
}

pub struct Array2DIterator<T>(std::vec::IntoIter<T>);

impl<T> Iterator for Array2DIterator<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}
