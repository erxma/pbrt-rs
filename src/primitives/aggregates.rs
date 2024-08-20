use crate::{geometry::Bounds3f, parallel, primitives::Primitive};

use super::PrimitiveEnum;

/// Aggregate primitives based on a bounding volume hierarchy (BVH).
pub struct BVHAggregate {
    prims: Vec<PrimitiveEnum>,
    nodes: Vec<LinearBVHNode>,
}

impl BVHAggregate {
    pub fn new(
        mut prims: Vec<PrimitiveEnum>,
        max_prims_in_node: usize,
        split_method: BVHSplitMethod,
    ) -> Self {
        // Cap prims per node to 255
        let max_prims_in_node = max_prims_in_node.min(255);

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
        max_prims_in_node: usize,
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
            let node = BVHBuildNode::new_leaf(first_prim_offset, prims.len(), bounds);
            BVHBuildResult {
                node: Some(node),
                n_nodes: 1,
            }
        };

        // If only one primitive, then recursion has bottomed out, create a leaf
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
                match Self::split_sah(prims_slice, centroid_bounds, max_prims_in_node) {
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
        let node = BVHBuildNode::new_interior(dim, left_result.node, right_result.node);

        // Done
        BVHBuildResult {
            node: Some(node),
            n_nodes,
        }
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

    fn split_sah(
        prims_slice: &mut [PrimitiveEnum],
        centroid_bounds: Bounds3f,
        max_prims_in_node: usize,
    ) -> Option<usize> {
        todo!()
    }

    fn build_hlbvh(prims: &mut [PrimitiveEnum]) -> BVHBuildResult {
        todo!()
    }

    fn flatten_bvh(build_result: BVHBuildResult) -> Vec<LinearBVHNode> {
        todo!()
    }
}

struct BVHBuildResult {
    node: Option<BVHBuildNode>,
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
        n_primitives: usize,
    },
    Interior {
        bounds: Bounds3f,
        left: Option<Box<BVHBuildNode>>,
        right: Option<Box<BVHBuildNode>>,
        split_axis: usize,
    },
}

impl BVHBuildNode {
    pub fn new_leaf(first_prim_offset: usize, n_primitives: usize, bounds: Bounds3f) -> Self {
        Self::Leaf {
            bounds,
            first_prim_offset,
            n_primitives,
        }
    }

    pub fn new_interior(
        split_axis: usize,
        left: Option<BVHBuildNode>,
        right: Option<BVHBuildNode>,
    ) -> Self {
        assert!(
            left.is_some() || right.is_some(),
            "Interior node should have at least one child"
        );

        let left_bounds = left.as_ref().map_or(Bounds3f::EMPTY, |node| node.bounds());
        let right_bounds = right.as_ref().map_or(Bounds3f::EMPTY, |node| node.bounds());

        Self::Interior {
            bounds: left_bounds.union(right_bounds),
            left: left.map(Box::new),
            right: right.map(Box::new),
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

struct LinearBVHNode {}
