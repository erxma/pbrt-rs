use std::ops::IndexMut;

pub trait Tuple<const N: usize, T>: IndexMut<usize, Output = T> {
    fn from_array(vals: [T; N]) -> Self;
}
