mod array2d;
mod bounds;
pub mod constants;
mod float;
mod frame;
mod interaction;
mod interval;
mod normal3;
mod point;
mod ray;
mod routines;
mod spherical;
mod square_matrix;
mod transform;
mod tuple;
mod vec;
mod vec_math;

pub mod common {
    pub use super::bounds::{Bounds2f, Bounds2i, Bounds3f, Bounds3i};
    pub use super::float::Float;
    pub use super::interval::Interval;
    pub use super::normal3::Normal3f;
    pub use super::point::{
        Point2Isize, Point2Usize, Point2f, Point2i, Point3f, Point3fi, Point3i,
    };
    pub use super::ray::{Ray, RayDifferential};
    pub use super::transform::Transform;
    pub use super::vec::{
        Vec2B, Vec2Isize, Vec2Usize, Vec2f, Vec2i, Vec3B, Vec3Isize, Vec3Usize, Vec3f, Vec3fi,
        Vec3i,
    };
}
pub use common::*;

pub use array2d::Array2D;
pub use float::{exponent, next_float_down, next_float_up, CompensatedFloat};
pub use frame::{Frame, FrameTransform};
pub use interaction::{
    IntraMediumInteraction, MediumInteraction, MediumInterfaceInteraction, SampleInteraction,
    Shading, SurfaceInteraction, SurfaceInteractionParams,
};
pub use ray::Differentials;
pub use routines::{erf, erf_inv, erff, fast_exp, gamma, lerp, safe_acos, safe_asin, safe_sqrt};
pub use spherical::{
    equal_area_sphere_to_square, equal_area_square_to_sphere, wrap_equal_area_square,
    DirectionCone, OctahedralVec,
};
pub use square_matrix::SquareMatrix;
pub use tuple::Tuple;
pub(crate) use tuple::{impl_tuple_math_ops, impl_tuple_math_ops_generic, TupleElement};
pub use vec_math::{spherical_direction, spherical_quad_area, spherical_triangle_area};
