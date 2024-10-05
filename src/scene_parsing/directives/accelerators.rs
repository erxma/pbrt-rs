use crate::{
    core::Float,
    scene_parsing::common::{
        impl_from_entity, EntityDirective, FromEntity, ParseContext, PbrtParseError,
    },
};

#[derive(Clone, Debug, PartialEq)]
pub enum Accelerator {
    Bvh(BvhAggregate),
    KdTree(KdTreeAggregate),
}

impl Default for Accelerator {
    fn default() -> Self {
        Self::Bvh(BvhAggregate::default())
    }
}

impl FromEntity for Accelerator {
    fn from_entity(entity: EntityDirective, ctx: &ParseContext) -> Result<Self, PbrtParseError> {
        assert_eq!(entity.identifier, "Accelerator");

        match entity.subtype {
            "bvh" => BvhAggregate::from_entity(entity, ctx).map(Accelerator::Bvh),
            "kdtree" => KdTreeAggregate::from_entity(entity, ctx).map(Accelerator::KdTree),
            invalid_type => Err(PbrtParseError::UnrecognizedVariant {
                entity: "Accelerator".to_string(),
                variant_name: invalid_type.to_owned(),
            }),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BvhAggregate {
    max_node_prims: usize,
    split_method: String,
}

impl Default for BvhAggregate {
    fn default() -> Self {
        Self {
            max_node_prims: 4,
            split_method: "sah".to_string(),
        }
    }
}

impl_from_entity! {
    BvhAggregate,
    has_defaults {
        "maxnodeprims" => max_node_prims,
        "splitmethod" => split_method,
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct KdTreeAggregate {
    intersection_cost: usize,
    traversal_cost: usize,
    empty_bonus: Float,
    max_prims: usize,
    max_depth: Option<usize>,
}

impl Default for KdTreeAggregate {
    fn default() -> Self {
        Self {
            intersection_cost: 5,
            traversal_cost: 1,
            empty_bonus: 0.5,
            max_prims: 1,
            max_depth: None,
        }
    }
}

impl_from_entity! {
    KdTreeAggregate,
    has_defaults {
        "intersectioncost" => intersection_cost,
        "traversalcost" => traversal_cost,
        "emptybonus" => empty_bonus,
        "maxprims" => max_prims,
        "maxdepth" => max_depth,
    }
}
