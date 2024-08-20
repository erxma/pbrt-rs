use itertools::iproduct;
use rayon::prelude::*;

use crate::{geometry::Bounds2i, math::Point2i, Float};

pub fn parallel_for_2d_with<T>(
    extent: Bounds2i,
    init: T,
    op: impl (Fn(&mut T, Bounds2i)) + Send + Sync,
) where
    T: Send + Clone,
{
    let est_tile_size = (extent.diagonal().x() * extent.diagonal().y()) as usize
        / (8 * rayon::current_num_threads());
    let tile_size = ((est_tile_size as Float).sqrt() as usize).clamp(1, 32);

    let tiles = iproduct!(
        (extent.p_min.x()..=extent.p_max.x()).step_by(tile_size),
        (extent.p_min.y()..=extent.p_max.y()).step_by(tile_size),
    )
    .map(|(x_start, y_start)| {
        let tile_size = tile_size as i32;
        let tile_start = Point2i::new(x_start, y_start);
        let tile_end = Point2i::new(
            (x_start + tile_size - 1).min(extent.p_max.x()),
            (y_start + tile_size - 1).min(extent.p_max.y()),
        );

        Bounds2i::new(tile_start, tile_end)
    });

    // TODO: Confirm that par_bridge is sufficient; could collect into Vec first?
    tiles.par_bridge().for_each_with(init, op);
}

pub fn join<A, B, ReturnA, ReturnB>(oper_a: A, oper_b: B) -> (ReturnA, ReturnB)
where
    A: FnOnce() -> ReturnA + Send,
    B: FnOnce() -> ReturnB + Send,
    ReturnA: Send,
    ReturnB: Send,
{
    rayon::join(oper_a, oper_b)
}
