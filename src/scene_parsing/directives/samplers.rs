use crate::scene_parsing::common::{
    impl_from_entity, EntityDirective, FromEntity, GraphicsState, PbrtParseError,
};

#[derive(Clone, Debug)]
pub enum Sampler {
    Independent(IndependentSampler),
}

impl Default for Sampler {
    fn default() -> Self {
        // FIXME: Should be ZSobol once it's available
        Self::Independent(IndependentSampler::default())
    }
}

impl FromEntity for Sampler {
    fn from_entity(entity: EntityDirective, state: &GraphicsState) -> Result<Self, PbrtParseError> {
        assert_eq!(entity.identifier, "Sampler");

        match entity.subtype {
            "independent" => {
                IndependentSampler::from_entity(entity, state).map(Sampler::Independent)
            }
            invalid_type => Err(PbrtParseError::UnrecognizedVariant {
                entity: "Sampler".to_string(),
                variant_name: invalid_type.to_owned(),
            }),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct IndependentSampler {
    pub seed: u64,
    pub pixel_samples: usize,
}

impl Default for IndependentSampler {
    fn default() -> Self {
        Self {
            seed: 0,
            pixel_samples: 16,
        }
    }
}

impl_from_entity! {
    IndependentSampler,
    has_defaults {
        "seed" => seed,
        "pixelsamples" => pixel_samples
    }
}
