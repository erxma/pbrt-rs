use crate::{geometry::Bounds3f, parallel, primitives::Primitive, Float};

use super::PrimitiveEnum;

/// Aggregate primitives based on a bounding volume hierarchy (BVH).
pub struct BVHAggregate {
    prims: Vec<PrimitiveEnum>,
    nodes: Vec<LinearBVHNode>,
}

impl BVHAggregate {
    pub fn new(
        mut prims: Vec<PrimitiveEnum>,
        max_prims_in_node: u8,
        split_method: BVHSplitMethod,
    ) -> Self {
        // TODO: Consider if it'd be worth it to precompute vec of centroids

        // Build BVH using given method
        let root_result = match split_method {
            BVHSplitMethod::HLBVH => Self::build_hlbvh(&mut prims),
            _ => Self::build_recursive(&mut prims, split_method, max_prims_in_node, 0),
        };

        // Convert BVH into compact representation in nodes array
        let nodes = Self::flatten_bvh(root_result);

        Self { prims, nodes }
    }

    fn build_recursive(
        prims_slice: &mut [PrimitiveEnum],
        split_method: BVHSplitMethod,
        max_prims_in_node: u8,
        first_prim_offset: usize,
    ) -> BVHBuildResult {
        // If number of prims to split up is greater than this number,
        // run the two subtasks in paralell. Otherwise just run in sequence.
        const MIN_PRIMS_TO_SPLIT_PARALLEL: usize = 128 * 1024;

        // Compute bounds containing all primitives in node
        let bounds = prims_slice
            .iter()
            .map(|prim| prim.bounds())
            .reduce(|total, bounds| total.union(bounds))
            .unwrap();

        // Creates a leaf from all the prims in this slice. Used in multiple cases below.
        // Takes prims as param so that it's not immediately borrowed here.
        let create_leaf = |prims: &[PrimitiveEnum]| {
            let node = BVHBuildNode::new_leaf(first_prim_offset, prims.len() as u8, bounds);
            BVHBuildResult { node, n_nodes: 1 }
        };

        // If only one primitive, then recursion has bottomed out, create a leaf
        // Also do so if bounds has no surface area (this also covers the 0 prims case)
        if bounds.surface_area() == 0.0 || prims_slice.len() == 1 {
            return create_leaf(prims_slice);
        }

        // Bounds containing all prim centroids
        let centroid_bounds = prims_slice
            .iter()
            .map(|prim| prim.bounds().centroid())
            .fold(Bounds3f::EMPTY, |total, centroid| {
                total.union_point(centroid)
            });
        // Choose the widest dimension as splitting axis
        let dim = centroid_bounds.max_extent();

        // In the rare case of all centroid points in the same position
        // (bounds has no volume), create a leaf.
        if centroid_bounds.p_max[dim] == centroid_bounds.p_min[dim] {
            return create_leaf(prims_slice);
        }

        // Partition prims into two based on split method
        let mid = match split_method {
            BVHSplitMethod::SAH => {
                // SAH may give a mid or decide to just create a leaf
                match Self::split_sah(prims_slice, centroid_bounds, max_prims_in_node, dim, bounds)
                {
                    Some(mid) => mid,
                    // Create a leaf as it has lower cost
                    None => {
                        return create_leaf(prims_slice);
                    }
                }
            }
            // Middle method  may fail, in which case fall back to EqualCounts
            BVHSplitMethod::Middle => Self::split_middle(prims_slice, centroid_bounds, dim)
                .unwrap_or_else(|| Self::split_equal_counts(prims_slice, dim)),
            BVHSplitMethod::EqualCounts => Self::split_equal_counts(prims_slice, dim),
            BVHSplitMethod::HLBVH => unreachable!(),
        };

        // Recursively build BVHs for children
        let num_prims = prims_slice.len();
        let (left_prims, right_prims) = prims_slice.split_at_mut(mid);
        let mut build_left = || {
            Self::build_recursive(
                left_prims,
                split_method,
                max_prims_in_node,
                first_prim_offset,
            )
        };
        let mut build_right = || {
            Self::build_recursive(
                right_prims,
                split_method,
                max_prims_in_node,
                first_prim_offset + mid,
            )
        };
        let (left_result, right_result) = if num_prims > MIN_PRIMS_TO_SPLIT_PARALLEL {
            // If number is large, do so in parallel
            parallel::join(build_left, build_right)
        } else {
            // Otherwise just do in series
            (build_left(), build_right())
        };

        // Tally up nodes (children's + self)
        let n_nodes = left_result.n_nodes + right_result.n_nodes + 1;
        // Create interior node with the two children
        let node = BVHBuildNode::new_interior(dim as u8, left_result.node, right_result.node);

        // Done
        BVHBuildResult { node, n_nodes }
    }

    /// Split by the midpoint of the primitives' centroids along the splitting axis,
    /// one group above and one group below.
    ///
    /// May fail to split into two groups, in which returns `None`.
    fn split_middle(
        prims_slice: &mut [PrimitiveEnum],
        centroid_bounds: Bounds3f,
        dim: usize,
    ) -> Option<usize> {
        let num_prims = prims_slice.len();

        // Midpoint of centroids along splitting axis
        let dim_mid = (centroid_bounds.p_min[dim] + centroid_bounds.p_max[dim]) / 2.0;
        // Partition by whether centroid is less than midpoint along axis
        // Would use partition_in_place if stable
        let mid = itertools::partition(prims_slice, |prim| prim.bounds().centroid()[dim] < dim_mid);

        // Return idx only if it's not at start or past end
        if (1..num_prims).contains(&mid) {
            Some(mid)
        } else {
            None
        }
    }

    /// Split into even halves, based on the primitives' centroid coords
    /// along the splitting axis, one group with smaller values and one group larger.
    fn split_equal_counts(prims_slice: &mut [PrimitiveEnum], dim: usize) -> usize {
        let mid = prims_slice.len() / 2;
        prims_slice.select_nth_unstable_by(mid, |prim_a, prim_b| {
            prim_a.bounds().centroid()[dim]
                .partial_cmp(&prim_b.bounds().centroid()[dim])
                .unwrap()
        });
        mid
    }

    /// Split by Surface Area Heuristic.
    fn split_sah(
        prims_slice: &mut [PrimitiveEnum],
        centroid_bounds: Bounds3f,
        max_prims_in_node: u8,
        dim: usize,
        bounds: Bounds3f,
    ) -> Option<usize> {
        const NUM_BUCKETS: usize = 12;
        const NUM_SPLITS: usize = NUM_BUCKETS - 1;

        // If down to two prims or less, just partition into halves
        if prims_slice.len() <= 2 {
            let mid = prims_slice.len() / 2;
            prims_slice.select_nth_unstable_by(mid, |prim_a, prim_b| {
                prim_a.bounds().centroid()[dim]
                    .partial_cmp(&prim_b.bounds().centroid()[dim])
                    .unwrap()
            });
            Some(mid)
        } else {
            // Initialize buckets
            let mut buckets: [BVHSplitBucket; NUM_BUCKETS] = Default::default();
            // Each bucket covers a even slice of the centroid bounds along the axis.
            // Find the position for the prim's centroid.
            let get_bucket_idx = |prim: &PrimitiveEnum| {
                let bucket_offset =
                    NUM_BUCKETS as Float * centroid_bounds.offset(prim.bounds().centroid())[dim];
                (bucket_offset as usize).min(NUM_BUCKETS)
            };
            // Contribute each prim to its bucket, as determined by above
            for prim in prims_slice.iter() {
                let b_idx = get_bucket_idx(prim);
                buckets[b_idx].count += 1;
                buckets[b_idx].bounds = buckets[b_idx].bounds.union(prim.bounds());
            }

            // Compute costs for splitting after each bucket:
            let mut costs = [0.0; NUM_SPLITS];

            // Partially initialize costs using a forward scan over splits
            // Num of prims below the current split
            let mut count_below = 0;
            // Bounds of those prims
            let mut bound_below = Bounds3f::EMPTY;
            for i in 0..NUM_SPLITS {
                bound_below = bound_below.union(buckets[i].bounds);
                count_below += buckets[i].count;
                costs[i] += count_below as Float * bound_below.surface_area();
            }

            // Similar to above but backwards
            let mut count_above = 0;
            let mut bound_above = Bounds3f::EMPTY;
            for i in (1..=NUM_SPLITS).rev() {
                bound_above = bound_above.union(buckets[i].bounds);
                count_above += buckets[i].count;
                costs[i - 1] += count_above as Float * bound_above.surface_area();
            }

            // Find bucket to split at that minimizes SAH metric
            let (min_cost_split_bucket, min_bucket_cost) = costs
                .iter()
                .enumerate()
                .min_by(|(_, cost_a), (_, cost_b)| cost_a.partial_cmp(cost_b).unwrap())
                .unwrap();

            // Compute leaf cost and SAH split cost for chosen split
            let leaf_cost = prims_slice.len() as Float;
            let min_cost = 0.5 + min_bucket_cost / bounds.surface_area();

            // Choose to split at selected bucket if cost is lower than leaf cost,
            // or if a leaf would have too many prims.
            // Otherwise choose to make leaf.
            if min_cost < leaf_cost || prims_slice.len() > max_prims_in_node.into() {
                let mid = itertools::partition(prims_slice, |prim| {
                    get_bucket_idx(prim) <= min_cost_split_bucket
                });
                Some(mid)
            } else {
                None
            }
        }
    }

    fn build_hlbvh(prims: &mut [PrimitiveEnum]) -> BVHBuildResult {
        todo!()
    }

    fn flatten_bvh(root_build_result: BVHBuildResult) -> Vec<LinearBVHNode> {
        // Prereserve vec to hold the total num of nodes
        let mut nodes = Vec::with_capacity(root_build_result.n_nodes);
        // Fill in the vec starting at the root
        Self::flatten_bvh_node(root_build_result.node, &mut nodes);
        nodes
    }

    fn flatten_bvh_node(build_node: BVHBuildNode, linear_nodes: &mut Vec<LinearBVHNode>) {
        match build_node {
            BVHBuildNode::Leaf {
                bounds,
                first_prim_offset,
                n_primitives,
            } => {
                let node = LinearBVHNode::Leaf {
                    bounds,
                    prims_offset: first_prim_offset,
                    num_prims: n_primitives,
                };
                linear_nodes.push(node);
            }
            BVHBuildNode::Interior {
                bounds,
                left,
                right,
                split_axis,
            } => {
                Self::flatten_bvh_node(*left, linear_nodes);
                // Current len of vec = the starting index of right child's nodes, record it
                let second_child_offset = linear_nodes.len();
                Self::flatten_bvh_node(*right, linear_nodes);
                let node = LinearBVHNode::Interior {
                    bounds,
                    second_child_offset,
                    axis: split_axis,
                };
                linear_nodes.push(node);
            }
        }
    }
}

struct BVHBuildResult {
    node: BVHBuildNode,
    n_nodes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BVHSplitMethod {
    /// Build a Hierarchical Linear Bounding Volume Hierarchy.
    HLBVH,
    /// Split by Surface Area Heuristic.
    SAH,
    /// Split by the midpoint of the primitives' centroids along the splitting axis,
    /// one group above and one group below.
    Middle,
    /// Split into even halves, based on the primitives' centroid coords
    /// along the splitting axis, one group with smaller values and one group larger.
    EqualCounts,
}

enum BVHBuildNode {
    Leaf {
        bounds: Bounds3f,
        first_prim_offset: usize,
        n_primitives: u8,
    },
    Interior {
        bounds: Bounds3f,
        left: Box<BVHBuildNode>,
        right: Box<BVHBuildNode>,
        split_axis: u8,
    },
}

impl BVHBuildNode {
    pub fn new_leaf(first_prim_offset: usize, n_primitives: u8, bounds: Bounds3f) -> Self {
        Self::Leaf {
            bounds,
            first_prim_offset,
            n_primitives,
        }
    }

    pub fn new_interior(split_axis: u8, left: BVHBuildNode, right: BVHBuildNode) -> Self {
        Self::Interior {
            bounds: left.bounds().union(right.bounds()),
            left: Box::new(left),
            right: Box::new(right),
            split_axis,
        }
    }

    fn bounds(&self) -> Bounds3f {
        match self {
            BVHBuildNode::Leaf { bounds, .. } => *bounds,
            BVHBuildNode::Interior { bounds, .. } => *bounds,
        }
    }
}

#[repr(align(32))]
enum LinearBVHNode {
    Leaf {
        bounds: Bounds3f,
        prims_offset: usize,
        num_prims: u8,
    },
    Interior {
        bounds: Bounds3f,
        second_child_offset: usize,
        axis: u8,
    },
}

#[derive(Clone, Copy)]
struct BVHSplitBucket {
    count: usize,
    bounds: Bounds3f,
}

impl Default for BVHSplitBucket {
    fn default() -> Self {
        Self {
            count: 0,
            bounds: Bounds3f::EMPTY,
        }
    }
}
