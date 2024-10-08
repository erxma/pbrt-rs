use crate::scene_parsing::common::{
    impl_from_entity, EntityDirective, FromEntity, ParseContext, PbrtParseError,
};

#[derive(Clone, Debug)]
pub enum Integrator {
    RandomWalk(RandomWalkIntegrator),
    SimplePath(SimplePathIntegrator),
}

impl Default for Integrator {
    fn default() -> Self {
        // FIXME: Should be VolPath once it's available
        Self::RandomWalk(RandomWalkIntegrator::default())
    }
}

impl FromEntity for Integrator {
    fn from_entity(entity: EntityDirective, ctx: &ParseContext) -> Result<Self, PbrtParseError> {
        assert_eq!(entity.identifier, "Integrator");

        match entity.subtype {
            "randomwalk" => {
                RandomWalkIntegrator::from_entity(entity, ctx).map(Integrator::RandomWalk)
            }
            "simplepath" => {
                SimplePathIntegrator::from_entity(entity, ctx).map(Integrator::SimplePath)
            }
            invalid_type => Err(PbrtParseError::UnrecognizedVariant {
                entity: "Integrator".to_string(),
                variant_name: invalid_type.to_owned(),
            }),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RandomWalkIntegrator {
    pub max_depth: usize,
}

impl Default for RandomWalkIntegrator {
    fn default() -> Self {
        Self { max_depth: 5 }
    }
}

impl_from_entity! {
    RandomWalkIntegrator,
    has_defaults {
        "maxdepth" => max_depth,
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SimplePathIntegrator {
    pub max_depth: usize,
    pub sample_bsdf: bool,
    pub sample_lights: bool,
}

impl Default for SimplePathIntegrator {
    fn default() -> Self {
        Self {
            max_depth: 5,
            sample_bsdf: true,
            sample_lights: true,
        }
    }
}

impl_from_entity! {
    SimplePathIntegrator,
    has_defaults {
        "maxdepth" => max_depth,
        "samplebsdf" => sample_bsdf,
        "samplelights" => sample_lights,
    }
}
