mod array2d;
mod equal_area_mapping;
mod float_utility;
mod interval;
mod normal3;
mod octahedral_vec;
mod point;
mod routines;
mod square_matrix;
mod tuple;
mod vec;
mod vec_math;

pub use array2d::Array2D;
pub use equal_area_mapping::{
    equal_area_sphere_to_square, equal_area_square_to_sphere, wrap_equal_area_square,
};
pub use float_utility::{next_float_down, next_float_up, CompensatedFloat, ONE_MINUS_EPSILON};
pub use interval::Interval;
pub use normal3::Normal3f;
pub use octahedral_vec::OctahedralVec;
pub use point::{Point2f, Point2i, Point3f, Point3fi, Point3i};
pub use routines::{
    difference_of_products, encode_morton_3, erf, erf_inv, erff, evaluate_polynomial, fast_exp,
    gamma, gaussian, lerp, safe_acos, safe_asin, safe_sqrt,
};
pub use square_matrix::SquareMatrix;
pub use tuple::Tuple;
pub(crate) use tuple::{impl_tuple_math_ops, impl_tuple_math_ops_generic, TupleElement};
pub use vec::{Vec2f, Vec2i, Vec3B, Vec3Usize, Vec3f, Vec3fi, Vec3i};
pub use vec_math::{spherical_direction, spherical_quad_area, spherical_triangle_area};
