use std::{
    mem,
    sync::{Arc, Mutex},
};

use num_traits::NumCast;

use crate::{
    core::{encode_morton_3, Bounds3f, Float, Ray},
    parallel::{self, parallel_map, parallel_map_enumerate},
    primitives::Primitive,
    shapes::ShapeIntersection,
};

use super::PrimitiveEnum;

/// Aggregate primitives based on a bounding volume hierarchy (BVH).
pub struct BVHAggregate {
    prims: Vec<Arc<PrimitiveEnum>>,
    nodes: Vec<LinearBVHNode>,
}

impl BVHAggregate {
    pub fn new(
        mut prims: Vec<Arc<PrimitiveEnum>>,
        max_prims_in_node: u8,
        split_method: BVHSplitMethod,
    ) -> Self {
        // OPTIMIZATION: Consider if it'd be worth it to precompute vec of centroids

        // Build BVH using given method
        let root_result;
        match split_method {
            BVHSplitMethod::HLBVH => {
                (root_result, prims) = Self::build_hlbvh(prims, max_prims_in_node);
            }
            method => {
                root_result = Self::build_recursive(&mut prims, method, max_prims_in_node, 0);
            }
        }

        // Convert BVH into compact representation in nodes array
        let nodes = Self::flatten_bvh(root_result);

        Self { prims, nodes }
    }

    fn build_recursive(
        prims_slice: &mut [Arc<PrimitiveEnum>],
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
        let create_leaf = |prims: &[Arc<PrimitiveEnum>]| {
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
        prims_slice: &mut [Arc<PrimitiveEnum>],
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
    fn split_equal_counts(prims_slice: &mut [Arc<PrimitiveEnum>], dim: usize) -> usize {
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
        prims_slice: &mut [Arc<PrimitiveEnum>],
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

    fn build_hlbvh(
        prims: Vec<Arc<PrimitiveEnum>>,
        max_prims_in_node: u8,
    ) -> (BVHBuildResult, Vec<Arc<PrimitiveEnum>>) {
        // Compute bounding box of all centroids
        let bounds = prims
            .iter()
            .map(|prim| prim.bounds())
            .reduce(|total, bounds| total.union(bounds))
            .unwrap();

        // Compute Morton indices of prims
        let mut morton_prims = parallel_map_enumerate(&prims, |(idx, p)| {
            MortonPrimitive::new(bounds, p.as_ref(), idx)
        });

        // Radix sort by Morton code
        MortonPrimitive::radix_sort(&mut morton_prims);

        // Create LBVH treelets at bottom of BVH:

        // Find intervals of prims for each treelet

        let same_treelet = |prim_a: &MortonPrimitive, prim_b: &MortonPrimitive| {
            const MASK: u32 = 0b00111111111111000000000000000000;
            (prim_a.code & MASK) == (prim_b.code & MASK)
        };

        let morton_slices: Vec<_> = morton_prims.chunk_by(same_treelet).collect();

        const FIRST_BIT_IDX: isize = 29 - 12;
        let ordered_prims = Arc::new(Mutex::new(Vec::with_capacity(prims.len())));
        let mut built_treelets: Vec<_> = parallel_map(morton_slices, |treelet| {
            Self::emit_lbvh(
                treelet,
                &prims,
                ordered_prims.clone(),
                FIRST_BIT_IDX,
                max_prims_in_node,
            )
        });

        // Create and return SAH BVH from LBVH treelets
        let root_result = Self::build_upper_sah(&mut built_treelets);
        (root_result, prims)
    }

    fn emit_lbvh(
        morton_slice: &[MortonPrimitive],
        prims: &[Arc<PrimitiveEnum>],
        ordered_prims: Arc<Mutex<Vec<Arc<PrimitiveEnum>>>>,
        bit_idx: isize,
        max_prims_in_node: u8,
    ) -> BVHBuildResult {
        // OPTIMIZATION: The book preallocates a vec to hold nodes
        // that will be created(the number is bounded).
        // Not doing that for now as it was leading to a lot of tricky ownership issues
        // that may or may not have defeated the purpose anyway,
        // at least with the current overall workflow.

        let num_prims = morton_slice.len();

        if bit_idx == -1 || num_prims < max_prims_in_node.into() {
            // Create and return leaf node of LBVH treelet
            let mut treelet_ordered_prims: Vec<_> = morton_slice
                .iter()
                .map(|mp| prims[mp.prim_idx].clone())
                .collect();
            let bounds = treelet_ordered_prims
                .iter()
                .map(|prim| prim.bounds())
                .reduce(|total, bound| total.union(bound))
                .unwrap();
            let first_prim_offset;
            {
                let mut ordered_prims = ordered_prims.lock().unwrap();
                first_prim_offset = ordered_prims.len();
                ordered_prims.append(&mut treelet_ordered_prims);
            }
            let node = BVHBuildNode::new_leaf(first_prim_offset, num_prims as u8, bounds);
            BVHBuildResult { node, n_nodes: 1 }
        } else {
            let mask = 1 << bit_idx;
            // Advance to next subtree level if there is no LBVH split for this bit
            // (all lie on one side)
            if (morton_slice[0].code & mask) == (morton_slice[num_prims - 1].code & mask) {
                return Self::emit_lbvh(
                    morton_slice,
                    prims,
                    ordered_prims,
                    bit_idx - 1,
                    max_prims_in_node,
                );
            }

            // Find LBVH split point for this dimension
            // Because of above check, this can never past end
            let split_offset = morton_slice
                .partition_point(|prim| (prim.code & mask) == (morton_slice[0].code & mask));

            // Create and return interior LBVH node
            let left = Self::emit_lbvh(
                &morton_slice[..split_offset],
                prims,
                ordered_prims.clone(),
                bit_idx - 1,
                max_prims_in_node,
            );
            let right = Self::emit_lbvh(
                &morton_slice[split_offset..],
                prims,
                ordered_prims,
                bit_idx - 1,
                max_prims_in_node,
            );
            let axis = bit_idx % 3;
            let n_nodes = 1 + left.n_nodes + right.n_nodes;
            let node = BVHBuildNode::new_interior(axis as u8, left.node, right.node);
            BVHBuildResult { node, n_nodes }
        }
    }

    fn build_upper_sah(treelets: &mut [BVHBuildResult]) -> BVHBuildResult {
        const NUM_BUCKETS: usize = 12;
        const NUM_SPLITS: usize = NUM_BUCKETS - 1;

        // If down to one, just return it
        if treelets.len() == 1 {
            treelets[0].clone()
        } else {
            // Bounds of all centroids in this treelet
            let centroid_bounds = treelets
                .iter()
                .map(|t| t.node.bounds().centroid())
                .fold(Bounds3f::EMPTY, |total, centroid| {
                    total.union_point(centroid)
                });
            // Pick widest dim as splitting axis
            let dim = centroid_bounds.max_extent();

            // Initialize buckets
            let mut buckets: [BVHSplitBucket; NUM_BUCKETS] = Default::default();
            // Each bucket covers a even slice of the centroid bounds along the axis.
            // Find the position for the node's centroid.
            let get_bucket_idx = |t: &BVHBuildNode| {
                let bucket_offset =
                    NUM_BUCKETS as Float * centroid_bounds.offset(t.bounds().centroid())[dim];
                (bucket_offset as usize).min(NUM_BUCKETS)
            };
            // Contribute each node to its bucket, as determined by above
            for treelet in treelets.iter() {
                let b_idx = get_bucket_idx(&treelet.node);
                buckets[b_idx].count += 1;
                buckets[b_idx].bounds = buckets[b_idx].bounds.union(treelet.node.bounds());
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
            let (min_cost_split_bucket, _) = costs
                .iter()
                .enumerate()
                .min_by(|(_, cost_a), (_, cost_b)| cost_a.partial_cmp(cost_b).unwrap())
                .unwrap();

            // Choose to split at selected bucket if cost is lower than leaf cost,
            // or if a leaf would have too many prims.
            // Otherwise choose to make leaf.
            let mid = itertools::partition(&mut *treelets, |t| {
                get_bucket_idx(&t.node) <= min_cost_split_bucket
            });

            let left = Self::build_upper_sah(&mut treelets[..mid]);
            let right = Self::build_upper_sah(&mut treelets[mid..]);

            let node = BVHBuildNode::new_interior(dim as u8, left.node, right.node);
            let n_nodes = 1 + left.n_nodes + right.n_nodes;

            BVHBuildResult { node, n_nodes }
        }
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
                // Need to push the root node first,
                // set second_child_offset once it's known
                // TODO: Could maybe be a little cleaner
                let node = LinearBVHNode::Interior {
                    bounds,
                    second_child_offset: Default::default(),
                    axis: split_axis,
                };
                linear_nodes.push(node);

                // Recursively call for children, placing their nodes after this root
                let idx = linear_nodes.len() - 1;
                Self::flatten_bvh_node(*left, linear_nodes);
                // Current len of vec = the starting index of right child's nodes, record it
                let second_child_offset = linear_nodes.len();
                Self::flatten_bvh_node(*right, linear_nodes);

                // Set second_child_offset now that it's known
                linear_nodes[idx] = LinearBVHNode::Interior {
                    bounds,
                    second_child_offset,
                    axis: split_axis,
                };
            }
        }
    }
}

impl Primitive for BVHAggregate {
    fn bounds(&self) -> Bounds3f {
        self.nodes[0].bounds()
    }

    fn intersect<'a>(
        &'a self,
        ray: &'a Ray,
        t_max: Option<Float>,
    ) -> Option<ShapeIntersection<'a>> {
        if self.nodes.is_empty() {
            return None;
        }

        let mut t_max = t_max.unwrap_or(Float::INFINITY);

        let inv_dir = [1.0 / ray.dir.x(), 1.0 / ray.dir.y(), 1.0 / ray.dir.z()];
        let dir_is_neg = inv_dir.map(|v| v < 0.0);

        // Follow ray through BVH nodes to find prim intersections
        // Use depth first search
        let mut shape_intersection = None;
        let mut nodes_to_visit_indices = Vec::with_capacity(64);
        nodes_to_visit_indices.push(0);
        while let Some(idx) = nodes_to_visit_indices.pop() {
            let node = &self.nodes[idx];
            // Check against BVH node
            if node.bounds().intersect_p(ray, t_max).is_some() {
                match node {
                    LinearBVHNode::Leaf {
                        prims_offset,
                        num_prims,
                        ..
                    } => {
                        // Intersect ray with primitives in leaf node
                        for prim in &self.prims[*prims_offset..*prims_offset + *num_prims as usize]
                        {
                            if let Some(prim_si) = prim.intersect(ray, Some(t_max)) {
                                // Update t_max to the hit point
                                t_max = prim_si.t_hit;
                                shape_intersection = Some(prim_si);
                            }
                        }
                    }
                    LinearBVHNode::Interior {
                        second_child_offset,
                        axis,
                        ..
                    } => {
                        // Pick order to check children
                        // Nodes structure was built such that First child after node
                        // is farther (greater value along split axis),
                        // so pick based on whether ray dir along axis is neg
                        if dir_is_neg[*axis as usize] {
                            // Try second child (farther) first
                            nodes_to_visit_indices.push(idx + 1);
                            nodes_to_visit_indices.push(*second_child_offset);
                        } else {
                            // Try first child (closer) first
                            nodes_to_visit_indices.push(*second_child_offset);
                            nodes_to_visit_indices.push(idx + 1);
                        }
                    }
                }
            }
        }

        shape_intersection
    }

    fn intersect_p(&self, ray: &Ray, t_max: Option<Float>) -> bool {
        let t_max = t_max.unwrap_or(Float::INFINITY);

        let inv_dir = [1.0 / ray.dir.x(), 1.0 / ray.dir.y(), 1.0 / ray.dir.z()];
        let dir_is_neg = inv_dir.map(|v| v < 0.0);

        // Follow ray through BVH nodes to find any intersection
        // Use depth first search
        let mut nodes_to_visit_indices = Vec::with_capacity(64);
        nodes_to_visit_indices.push(0);
        while let Some(idx) = nodes_to_visit_indices.pop() {
            let node = &self.nodes[idx];
            // Check against BVH node
            if node.bounds().intersect_p(ray, t_max).is_some() {
                match node {
                    LinearBVHNode::Leaf {
                        prims_offset,
                        num_prims,
                        ..
                    } => {
                        // Intersect ray with primitives in leaf node
                        for prim in &self.prims[*prims_offset..*prims_offset + *num_prims as usize]
                        {
                            // Intersection found, return true
                            if prim.intersect_p(ray, Some(t_max)) {
                                return true;
                            }
                        }
                    }
                    LinearBVHNode::Interior {
                        second_child_offset,
                        axis,
                        ..
                    } => {
                        // Pick order to check children
                        // Nodes structure was built such that First child after node
                        // is farther (greater value along split axis),
                        // so pick based on whether ray dir along axis is neg
                        if dir_is_neg[*axis as usize] {
                            // Try second child (farther) first
                            nodes_to_visit_indices.push(idx + 1);
                            nodes_to_visit_indices.push(*second_child_offset);
                        } else {
                            // Try first child (closer) first
                            nodes_to_visit_indices.push(*second_child_offset);
                            nodes_to_visit_indices.push(idx + 1);
                        }
                    }
                }
            }
        }

        false
    }
}

#[derive(Clone)]
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

#[derive(Clone)]
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
#[derive(Debug)]
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

impl LinearBVHNode {
    fn bounds(&self) -> Bounds3f {
        match self {
            LinearBVHNode::Leaf { bounds, .. } => *bounds,
            LinearBVHNode::Interior { bounds, .. } => *bounds,
        }
    }
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

#[derive(Clone, Copy)]
struct MortonPrimitive {
    prim_idx: usize,
    code: u32,
}

impl MortonPrimitive {
    const MORTON_BITS: usize = 10;
    const MORTON_SCALE: usize = 1 << Self::MORTON_BITS;

    pub fn new(bounds: Bounds3f, prim: &impl Primitive, prim_idx: usize) -> Self {
        let centroid_offset = bounds.offset(prim.bounds().centroid());
        let offset = centroid_offset * Self::MORTON_SCALE as Float;
        let code = encode_morton_3(
            NumCast::from(offset.x()).unwrap(),
            NumCast::from(offset.x()).unwrap(),
            NumCast::from(offset.x()).unwrap(),
        );
        Self { prim_idx, code }
    }

    pub fn radix_sort(values: &mut [MortonPrimitive]) {
        const BITS_PER_PASS: usize = 6;
        const NUM_BITS: usize = 30;
        const NUM_PASSES: usize = NUM_BITS / BITS_PER_PASS;
        const NUM_BUCKETS: usize = 1 << BITS_PER_PASS;
        const BITMASK: u32 = (1 << BITS_PER_PASS) - 1;

        let mut temp = values.to_owned();
        for pass in 0..NUM_PASSES {
            // Perform one pass of radix sort:
            let low_bit = pass * BITS_PER_PASS;
            // Set in and out slices for pass
            let (in_slice, out_slice) = if pass % 2 == 0 {
                (&mut *values, temp.as_mut_slice())
            } else {
                (temp.as_mut_slice(), &mut *values)
            };

            // Count num of zero bits in in slice for current sort bit
            let mut bucket_tallies = [0; NUM_BUCKETS];
            for mp in in_slice.iter() {
                let bucket = (mp.code >> low_bit) & BITMASK;
                bucket_tallies[bucket as usize] += 1;
            }

            // Compute starting idx in output slice for each bucket
            let mut out_indices = [0; NUM_BUCKETS];
            for i in 1..NUM_BUCKETS {
                out_indices[i] = out_indices[i - 1] + bucket_tallies[i - 1];
            }

            // Store sorted values in out slice
            for mp in in_slice {
                let bucket = (mp.code >> low_bit) & BITMASK;
                mem::swap(mp, &mut out_slice[out_indices[bucket as usize]])
            }
        }

        // Once done, make sure that the passed values has the final result
        if NUM_PASSES % 2 == 1 {
            values.copy_from_slice(&temp);
        }
    }
}
