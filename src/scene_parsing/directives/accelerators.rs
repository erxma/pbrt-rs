use crate::{
    core::Float,
    primitives::BVHSplitMethod,
    scene_parsing::common::{
        impl_from_entity, EntityDirective, FromEntity, GraphicsState, PbrtParseError, Value,
    },
};

#[derive(Clone, Debug, PartialEq)]
pub enum AcceleratorDesc {
    Bvh(BvhAggregate),
    KdTree(KdTreeAggregate),
}

impl Default for AcceleratorDesc {
    fn default() -> Self {
        Self::Bvh(BvhAggregate::default())
    }
}

impl FromEntity for AcceleratorDesc {
    fn from_entity(entity: EntityDirective, state: &GraphicsState) -> Result<Self, PbrtParseError> {
        assert_eq!(entity.identifier, "Accelerator");

        match entity.subtype {
            "bvh" => BvhAggregate::from_entity(entity, state).map(AcceleratorDesc::Bvh),
            "kdtree" => KdTreeAggregate::from_entity(entity, state).map(AcceleratorDesc::KdTree),
            invalid_type => Err(PbrtParseError::UnrecognizedVariant {
                entity: "Accelerator".to_string(),
                variant_name: invalid_type.to_owned(),
            }),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BvhAggregate {
    pub max_node_prims: u8,
    pub split_method: BVHSplitMethod,
}

impl Default for BvhAggregate {
    fn default() -> Self {
        Self {
            max_node_prims: 4,
            split_method: BVHSplitMethod::SAH,
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
    pub intersection_cost: usize,
    pub traversal_cost: usize,
    pub empty_bonus: Float,
    pub max_prims: usize,
    pub max_depth: Option<usize>,
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

impl TryFrom<Value> for BVHSplitMethod {
    type Error = PbrtParseError;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        let value = String::try_from(value)?;

        let method = match value.as_str() {
            "sah" => Self::SAH,
            "middle" => Self::Middle,
            "equal" => Self::EqualCounts,
            "hlbvh" => Self::HLBVH,
            _ => {
                return Err(PbrtParseError::InvalidValue {
                    expected: "sah, middle, equal, or hlbvh".to_string(),
                    found: Value::String(value),
                })
            }
        };

        Ok(method)
    }
}
