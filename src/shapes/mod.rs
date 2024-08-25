mod base;
mod bilinear_patch;
mod sphere;

pub use base::{
    QuadricIntersection, Shape, ShapeEnum, ShapeIntersection, ShapeSample, ShapeSampleContext,
};
pub use bilinear_patch::{
    intersect_bilinear_patch, BilinearIntersection, BilinearPatch, BilinearPatchMesh,
};
pub use sphere::Sphere;
