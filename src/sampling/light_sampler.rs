use std::sync::Arc;

use crate::{
    core::Float,
    lights::{Light, LightEnum, LightSampleContext},
};

pub trait LightSampler {
    fn sample(&self, u: Float) -> Option<SampledLight>;
    fn pmf(&self, light: &impl Light) -> Float;
    fn sample_with_context(&self, ctx: &LightSampleContext, u: Float) -> Option<SampledLight>;
    fn pmf_with_context(&self, ctx: &LightSampleContext, light: &impl Light) -> Float;
}

pub struct SampledLight {
    pub light: Arc<LightEnum>,
    pub prob: Float,
}

pub struct UniformLightSampler {
    lights: Vec<Arc<LightEnum>>,
}

impl UniformLightSampler {
    pub fn new(lights: &[Arc<LightEnum>]) -> Self {
        Self {
            lights: lights.to_owned(),
        }
    }
}

impl LightSampler for UniformLightSampler {
    fn sample(&self, u: Float) -> Option<SampledLight> {
        if self.lights.is_empty() {
            return None;
        }

        let num_lights = self.lights.len();
        // Map sample uniformly to range of light indices (num lights)
        let light_idx = ((u * num_lights as Float) as usize).min(num_lights - 1);
        let light = self.lights[light_idx].clone();
        let prob = 1.0 / num_lights as Float;
        Some(SampledLight { light, prob })
    }

    fn pmf(&self, _light: &impl Light) -> Float {
        if self.lights.is_empty() {
            return 0.0;
        }

        // Probability is always uniform
        1.0 / self.lights.len() as Float
    }

    fn sample_with_context(&self, _ctx: &LightSampleContext, u: Float) -> Option<SampledLight> {
        self.sample(u)
    }

    fn pmf_with_context(&self, _ctx: &LightSampleContext, light: &impl Light) -> Float {
        self.pmf(light)
    }
}
