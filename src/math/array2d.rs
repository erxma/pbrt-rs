use crate::geometry::bounds::Bounds2i;

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
