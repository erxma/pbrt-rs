mod array2d;
mod equal_area_mapping;
pub mod float_utility;
mod interval;
mod normal3;
mod octahedral_vec;
mod point;
mod routines;
mod square_matrix;
mod tuple;
mod vec;
pub mod vec_math;

pub use array2d::Array2D;
pub use equal_area_mapping::{
    equal_area_sphere_to_square, equal_area_square_to_sphere, wrap_equal_area_square,
};
pub use interval::Interval;
pub use normal3::Normal3f;
pub use octahedral_vec::OctahedralVec;
pub use point::{Point2f, Point2i, Point3f, Point3fi, Point3i};
pub use routines::{evaluate_polynomial, gamma, lerp, safe_acos, safe_asin, safe_sqrt};
pub use square_matrix::SquareMatrix;
pub use tuple::Tuple;
pub use vec::{Vec2f, Vec2i, Vec3B, Vec3Usize, Vec3f, Vec3fi, Vec3i};
