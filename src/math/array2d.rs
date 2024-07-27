use crate::geometry::bounds3::Bounds2i;

#[derive(Clone, Debug)]
pub struct Array2D<T> {
    extent: Bounds2i,
    values: Vec<T>,
}
