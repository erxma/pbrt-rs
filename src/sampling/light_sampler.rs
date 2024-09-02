use std::sync::Arc;

use crate::{
    lights::{Light, LightEnum, LightSampleContext},
    Float,
};

pub trait LightSampler {
    fn sample(u: Float) -> Option<SampledLight>;
    fn pmf(light: &impl Light) -> Float;
    fn sample_with_context(ctx: &LightSampleContext, u: Float) -> Option<SampledLight>;
    fn pmf_with_context(ctx: &LightSampleContext, light: &impl Light) -> Float;
}

pub struct SampledLight {
    pub light: Arc<LightEnum>,
    pub prob: Float,
}
