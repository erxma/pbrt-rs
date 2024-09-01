use std::{
    cell::UnsafeCell,
    cmp::min,
    fmt::{self, Display},
    ops::{Index, IndexMut},
    vec,
};

use itertools::Itertools;

use crate::geometry::Bounds2i;

use super::point::Point2i;

#[derive(Debug)]
pub struct Array2D<T> {
    extent: Bounds2i,
    values: UnsafeCell<Vec<T>>,
}

unsafe impl<T: Sync> Sync for Array2D<T> {}

impl<T> Array2D<T> {
    pub fn fill_default(extent: Bounds2i) -> Self
    where
        T: Default,
    {
        let n = extent.area() as usize;
        let mut values = Vec::new();
        values.resize_with(n, Default::default);
        Self {
            extent,
            values: UnsafeCell::new(values),
        }
    }

    pub fn fill_value(extent: Bounds2i, value: T) -> Self
    where
        T: Clone,
    {
        let n = extent.area() as usize;
        let values = vec![value; n];
        Self {
            extent,
            values: UnsafeCell::new(values),
        }
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

    /// Apply a mutation to a point in the array, requiring only an immutable receiver `&self`.
    /// This can be used to avoid taking an `&mut self` to the entire array to write
    /// to a point, allowing concurrent writes.
    ///
    /// # Safety
    /// Exclusivity on each point will be unenforced, so data races may occur
    /// if called on the same point concurrently.
    /// Therefore, the user is responsible for ensuring that
    /// each point is mutated by at most one thread at a time.
    pub unsafe fn mutate_unchecked(&self, p: Point2i, op: impl Fn(&mut T)) {
        let x = (p.x() - self.extent.p_min.x()) as usize;
        let y = (p.y() - self.extent.p_min.y()) as usize;
        let idx = y * self.x_size() + x;
        let val = unsafe { &mut (*self.values.get())[idx] };
        op(val);
    }

    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            arr: self,
            current_index: 0,
        }
    }

    pub fn tiles_iter_mut(&self, tile_width: usize, tile_height: usize) -> TilesMut<'_, T> {
        TilesMut {
            arr: self,
            current_x_start: self.extent.p_min.x(),
            current_y_start: self.extent.p_min.y(),
            tile_width,
            tile_height,
        }
    }

    fn index_linear(&self, index: usize) -> &T {
        debug_assert!(index < self.num_values(), "Index out of bounds for Array2D");
        unsafe { &(*self.values.get())[index] }
    }

    fn index_linear_mut(&mut self, index: usize) -> &mut T {
        debug_assert!(index < self.num_values(), "Index out of bounds for Array2D");
        unsafe { &mut (*self.values.get())[index] }
    }
}

impl<T> Index<Point2i> for Array2D<T> {
    type Output = T;

    fn index(&self, p: Point2i) -> &Self::Output {
        debug_assert!(p.inside_exclusive(self.extent));

        let x = (p.x() - self.extent.p_min.x()) as usize;
        let y = (p.y() - self.extent.p_min.y()) as usize;
        let idx = y * self.x_size() + x;

        self.index_linear(idx)
    }
}

impl<T> IndexMut<Point2i> for Array2D<T> {
    fn index_mut(&mut self, p: Point2i) -> &mut Self::Output {
        debug_assert!(p.inside_exclusive(self.extent));

        let x = (p.x() - self.extent.p_min.x()) as usize;
        let y = (p.y() - self.extent.p_min.y()) as usize;
        let idx = y * self.x_size() + x;

        self.index_linear_mut(idx)
    }
}

impl<T> Index<usize> for Array2D<T> {
    type Output = [T];

    fn index(&self, row: usize) -> &Self::Output {
        debug_assert!(row < self.y_size(), "Row index out of bounds for Array2D");
        let row_start = row * self.x_size();
        unsafe { &(*self.values.get())[row_start..row_start + self.x_size()] }
    }
}

impl<T> IndexMut<usize> for Array2D<T> {
    fn index_mut(&mut self, row: usize) -> &mut Self::Output {
        debug_assert!(row < self.y_size(), "Row index out of bounds for Array2D");
        let row_start = row * self.x_size();
        unsafe { &mut (*self.values.get())[row_start..row_start + self.x_size()] }
    }
}

impl<T: Display> fmt::Display for Array2D<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Use precision, if specified, for the values
        let fmt_val = |val| {
            if let Some(p) = f.precision() {
                format!("{:.*}", p, val)
            } else {
                format!("{}", val)
            }
        };
        let fmt_row = |row: usize| format!("[{}]", self[row].iter().map(fmt_val).join(", "));
        let mut rows = (0..self.y_size()).map(fmt_row);

        // If alternate (#) specified, pretty-print on separate rows.
        // Otherwise print on one row.
        if f.alternate() {
            // e.g.
            // [[0.0, 1.0, 2.0],
            //  [3.0, 4.0, 5.0]]
            write!(f, "[{}]", rows.join(",\n "))
        } else {
            // e.g.
            // [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
            write!(f, "[{}]", rows.join(", "))
        }
    }
}

pub struct Iter<'a, T> {
    arr: &'a Array2D<T>,
    current_index: usize,
}

// Iterator is effective same as the vec's,
// equivalent to going through each row, "left to right" within each row
impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index < self.arr.num_values() {
            let item = self.arr.index_linear(self.current_index);
            self.current_index += 1;
            Some(item)
        } else {
            None
        }
    }
}

pub struct TilesMut<'a, T> {
    arr: &'a Array2D<T>,
    current_x_start: i32,
    current_y_start: i32,
    tile_width: usize,
    tile_height: usize,
}

impl<'a, T> Iterator for TilesMut<'a, T> {
    type Item = Tile<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        // Past the last column. No more to produce.
        if self.current_y_start >= self.arr.extent.p_max.y() {
            return None;
        }

        let tile_x_end = min(
            self.current_x_start + self.tile_width as i32,
            self.arr.extent.p_max.x(),
        );
        let tile_y_end = min(
            self.current_y_start + self.tile_height as i32,
            self.arr.extent.p_max.y(),
        );
        let tile = Tile {
            arr: self.arr,
            extent: Bounds2i::new(
                Point2i::new(self.current_x_start, self.current_y_start),
                Point2i::new(tile_x_end, tile_y_end),
            ),
        };

        self.current_x_start = tile_x_end;
        if self.current_x_start >= self.arr.extent.p_max.x() {
            self.current_x_start = self.arr.extent.p_min.x();
            self.current_y_start += self.tile_height as i32;
        }

        Some(tile)
    }
}

pub struct Tile<'a, T> {
    arr: &'a Array2D<T>,
    extent: Bounds2i,
}

impl<'a, T> Tile<'a, T> {
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

impl<'a, T> Index<Point2i> for Tile<'a, T> {
    type Output = T;

    fn index(&self, p: Point2i) -> &Self::Output {
        assert!(p.inside_exclusive(self.extent));
        &self.arr[p]
    }
}

impl<'a, T> IndexMut<Point2i> for Tile<'a, T> {
    fn index_mut(&mut self, p: Point2i) -> &mut Self::Output {
        assert!(p.inside_exclusive(self.extent));

        let x = (p.x() - self.extent.p_min.x()) as usize;
        let y = (p.y() - self.extent.p_min.y()) as usize;
        let idx = y * self.x_size() + x;

        unsafe { &mut (*self.arr.values.get())[idx] }
    }
}
