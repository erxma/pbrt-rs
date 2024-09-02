use delegate::delegate;
use enum_dispatch::enum_dispatch;

use crate::{
    sampling::spectrum::{SampledSpectrum, SampledWavelengths, SpectrumEnum},
    Float,
};

use super::TextureEvalContext;

#[enum_dispatch]
#[derive(Debug)]
pub enum FloatTextureEnum {
    Constant(ConstantFloatTexture),
}

impl FloatTextureEnum {
    delegate! {
        #[through(FloatTexture)]
        to self {
            pub fn eval(&self, ctx: &TextureEvalContext) -> Float;
        }
    }
}

#[enum_dispatch(FloatTextureEnum)]
pub trait FloatTexture {
    fn eval(&self, ctx: &TextureEvalContext) -> Float;
}

#[enum_dispatch]
#[derive(Debug)]
pub enum SpectrumTextureEnum {
    Constant(ConstantSpectrumTexture),
}

impl SpectrumTextureEnum {
    delegate! {
        #[through(SpectrumTexture)]
        to self {
            pub fn eval(&self, ctx: &TextureEvalContext, lambda: &SampledWavelengths) -> SampledSpectrum;
        }
    }
}

#[enum_dispatch(SpectrumTextureEnum)]
pub trait SpectrumTexture {
    fn eval(&self, ctx: &TextureEvalContext, lambda: &SampledWavelengths) -> SampledSpectrum;
}

#[derive(Debug)]
pub struct ConstantFloatTexture {
    value: Float,
}

impl FloatTexture for ConstantFloatTexture {
    fn eval(&self, _ctx: &TextureEvalContext) -> Float {
        self.value
    }
}

impl ConstantFloatTexture {
    pub fn new(value: Float) -> Self {
        Self { value }
    }
}

#[derive(Debug)]
pub struct ConstantSpectrumTexture {
    value: SpectrumEnum,
}

impl SpectrumTexture for ConstantSpectrumTexture {
    fn eval(&self, _ctx: &TextureEvalContext, lambda: &SampledWavelengths) -> SampledSpectrum {
        self.value.sample(lambda)
    }
}

impl ConstantSpectrumTexture {
    pub fn new(value: SpectrumEnum) -> Self {
        Self { value }
    }
}
