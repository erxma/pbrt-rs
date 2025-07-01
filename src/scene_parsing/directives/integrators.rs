use crate::{
    integrators::LightSampleStrategy,
    scene_parsing::common::{
        impl_from_entity, EntityDirective, FromEntity, GraphicsState, PbrtParseError, Value,
    },
};

#[derive(Clone, Debug)]
pub enum Integrator {
    RandomWalk(RandomWalkIntegrator),
    SimplePath(SimplePathIntegrator),
    Path(PathIntegrator),
}

impl Default for Integrator {
    fn default() -> Self {
        // FIXME: Should be VolPath once it's available
        Self::RandomWalk(RandomWalkIntegrator::default())
    }
}

impl FromEntity for Integrator {
    fn from_entity(entity: EntityDirective, state: &GraphicsState) -> Result<Self, PbrtParseError> {
        assert_eq!(entity.identifier, "Integrator");

        match entity.subtype {
            "randomwalk" => {
                RandomWalkIntegrator::from_entity(entity, state).map(Integrator::RandomWalk)
            }
            "simplepath" => {
                SimplePathIntegrator::from_entity(entity, state).map(Integrator::SimplePath)
            }
            "path" => PathIntegrator::from_entity(entity, state).map(Integrator::Path),
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

#[derive(Clone, Debug, PartialEq)]
pub struct PathIntegrator {
    pub max_depth: usize,
    pub light_sampler: LightSampleStrategy,
    pub regularize: bool,
}

impl Default for PathIntegrator {
    fn default() -> Self {
        Self {
            max_depth: 5,
            light_sampler: LightSampleStrategy::Bvh,
            regularize: false,
        }
    }
}

impl_from_entity! {
    PathIntegrator,
    has_defaults {
        "maxdepth" => max_depth,
        "lightsampler" => light_sampler,
        "regularize" => regularize,
    }
}

impl TryFrom<Value> for LightSampleStrategy {
    type Error = PbrtParseError;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        let strategy = match String::try_from(value.clone())?.as_str() {
            "bvh" => Self::Bvh,
            "uniform" => Self::Uniform,
            "power" => Self::Power,
            _ => {
                return Err(PbrtParseError::InvalidValue {
                    expected: "bvh, uniform, or power".to_string(),
                    found: value,
                })
            }
        };

        Ok(strategy)
    }
}
