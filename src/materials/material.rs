use crate::{
    math::{Normal3f, Vec3f},
    reflection::DiffuseBxDF,
    sampling::spectrum::{SampledSpectrum, SampledWavelengths},
    Float,
};

use super::{FloatTexture, SpectrumTexture, SpectrumTextureEnum, TextureEvalContext};

pub enum MaterialEnum {
    Diffuse(DiffuseBxDF),
}

pub trait Material {
    type BxDF: crate::reflection::BxDF;

    fn bxdf(
        &self,
        tex_eval: &impl TextureEvaluator,
        ctx: &MaterialEvalContext,
        lambda: &SampledWavelengths,
    ) -> Self::BxDF;
}

#[derive(Clone, Debug)]
pub struct MaterialEvalContext {
    pub tex_eval_ctx: TextureEvalContext,
    pub wo: Vec3f,
    pub ns: Normal3f,
    pub dpdus: Vec3f,
}

pub trait TextureEvaluator {
    fn eval(&self, texture: &impl FloatTexture, ctx: &TextureEvalContext) -> Float;
    fn eval_spectrum(
        &self,
        texture: &impl SpectrumTexture,
        ctx: &TextureEvalContext,
        lambda: &SampledWavelengths,
    ) -> SampledSpectrum;
}

pub struct UniversalTextureEvaluator {}

impl TextureEvaluator for UniversalTextureEvaluator {
    fn eval(&self, texture: &impl FloatTexture, ctx: &TextureEvalContext) -> Float {
        texture.eval(ctx)
    }

    fn eval_spectrum(
        &self,
        texture: &impl SpectrumTexture,
        ctx: &TextureEvalContext,
        lambda: &SampledWavelengths,
    ) -> SampledSpectrum {
        texture.eval(ctx, lambda)
    }
}

pub struct DiffuseMaterial<'a> {
    reflectance: SpectrumTextureEnum<'a>,
}

impl<'a> DiffuseMaterial<'a> {
    pub fn new(reflectance: SpectrumTextureEnum<'a>) -> Self {
        Self { reflectance }
    }
}

impl Material for DiffuseMaterial<'_> {
    type BxDF = DiffuseBxDF;

    fn bxdf(
        &self,
        tex_eval: &impl TextureEvaluator,
        ctx: &MaterialEvalContext,
        lambda: &SampledWavelengths,
    ) -> Self::BxDF {
        let reflectance = tex_eval
            .eval_spectrum(&self.reflectance, &ctx.tex_eval_ctx, lambda)
            .clamp(0.0, 1.0);
        DiffuseBxDF::new(reflectance)
    }
}
