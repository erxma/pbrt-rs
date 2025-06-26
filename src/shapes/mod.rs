mod base;
mod bilinear_patch;
mod sphere;
mod tri_quad;
mod triangle;

pub use base::{
    QuadricIntersection, Shape, ShapeEnum, ShapeIntersection, ShapeSample, ShapeSampleContext,
};
pub use bilinear_patch::{
    intersect_bilinear_patch, BilinearIntersection, BilinearPatch, BilinearPatchMesh,
};
pub use sphere::Sphere;
pub use tri_quad::{FromPlyError, TriQuadMesh};
pub use triangle::{intersect_triangle, Triangle, TriangleIntersection, TriangleMesh};
