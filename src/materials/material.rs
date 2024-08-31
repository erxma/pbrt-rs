use std::{borrow::Cow, sync::Arc};

use crate::{
    math::{Normal3f, Vec3f},
    reflection::{DielectricBxDF, DiffuseBxDF, TrowbridgeReitz},
    sampling::spectrum::{SampledSpectrum, SampledWavelengths, SpectrumEnum},
    Float,
};

use super::{
    FloatTexture, FloatTextureEnum, SpectrumTexture, SpectrumTextureEnum, TextureEvalContext,
};

pub enum MaterialEnum {
    Diffuse(DiffuseMaterial),
    Dielectric(DielectricMaterial),
}

pub trait Material {
    type BxDF: crate::reflection::BxDF;

    fn bxdf(
        &self,
        tex_eval: &impl TextureEvaluator,
        ctx: &MaterialEvalContext,
        lambda: Cow<SampledWavelengths>,
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

pub struct DiffuseMaterial {
    reflectance: Arc<SpectrumTextureEnum>,
}

impl DiffuseMaterial {
    pub fn new(reflectance: Arc<SpectrumTextureEnum>) -> Self {
        Self { reflectance }
    }
}

impl Material for DiffuseMaterial {
    type BxDF = DiffuseBxDF;

    fn bxdf(
        &self,
        tex_eval: &impl TextureEvaluator,
        ctx: &MaterialEvalContext,
        lambda: Cow<SampledWavelengths>,
    ) -> Self::BxDF {
        let reflectance = tex_eval
            .eval_spectrum(&*self.reflectance, &ctx.tex_eval_ctx, &lambda)
            .clamp(0.0, 1.0);
        DiffuseBxDF::new(reflectance)
    }
}

impl From<DiffuseMaterial> for MaterialEnum {
    fn from(value: DiffuseMaterial) -> Self {
        Self::Diffuse(value)
    }
}

pub struct DielectricMaterial {
    u_roughness: Arc<FloatTextureEnum>,
    v_roughness: Arc<FloatTextureEnum>,
    remap_roughness: bool,
    eta: Arc<SpectrumEnum>,
}

impl DielectricMaterial {
    pub fn new(
        u_roughness: Arc<FloatTextureEnum>,
        v_roughness: Arc<FloatTextureEnum>,
        remap_roughness: bool,
        eta: Arc<SpectrumEnum>,
    ) -> Self {
        Self {
            u_roughness,
            v_roughness,
            remap_roughness,
            eta,
        }
    }
}

impl Material for DielectricMaterial {
    type BxDF = DielectricBxDF;

    fn bxdf(
        &self,
        tex_eval: &impl TextureEvaluator,
        ctx: &MaterialEvalContext,
        mut lambda: Cow<SampledWavelengths>,
    ) -> Self::BxDF {
        // Compute index of refraction
        let sampled_eta = self.eta.at(lambda[0]);
        if !self.eta.is_constant() {
            lambda.to_mut().terminate_secondary();
        }

        // Create microfacet distribution
        let mut u_rough = tex_eval.eval(&*self.u_roughness, &ctx.tex_eval_ctx);
        let mut v_rough = tex_eval.eval(&*self.v_roughness, &ctx.tex_eval_ctx);
        if self.remap_roughness {
            u_rough = TrowbridgeReitz::roughness_to_alpha(u_rough);
            v_rough = TrowbridgeReitz::roughness_to_alpha(v_rough);
        }
        let distrib = TrowbridgeReitz::new(u_rough, v_rough);

        // Final BxDF
        DielectricBxDF::new(sampled_eta, distrib)
    }
}

impl From<DielectricMaterial> for MaterialEnum {
    fn from(value: DielectricMaterial) -> Self {
        Self::Dielectric(value)
    }
}
