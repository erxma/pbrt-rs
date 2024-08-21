mod atomic;
mod loops;

pub use atomic::AtomicF64;
pub use loops::{join, parallel_for, parallel_for_2d_with, parallel_map, parallel_map_enumerate};
