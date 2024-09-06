use std::sync::Arc;

use crate::{
    geometry::SurfaceInteraction,
    math::{Normal3f, Vec3f},
    memory::ScratchBuffer,
    reflection::{BxDFEnum, DielectricBxDF, DiffuseBxDF, TrowbridgeReitz, BSDF},
    sampling::spectrum::{SampledSpectrum, SampledWavelengths, SpectrumEnum},
    Float,
};

use super::{
    FloatTexture, FloatTextureEnum, SpectrumTexture, SpectrumTextureEnum, TextureEvalContext,
};

#[derive(Debug)]
pub enum MaterialEnum {
    Diffuse(DiffuseMaterial),
    Dielectric(DielectricMaterial),
}

impl Material for MaterialEnum {
    type BxDF = BxDFEnum;

    fn bxdf(
        &self,
        tex_eval: &impl TextureEvaluator,
        ctx: &MaterialEvalContext,
        lambda: &mut SampledWavelengths,
    ) -> Self::BxDF {
        match self {
            MaterialEnum::Diffuse(m) => m.bxdf(tex_eval, ctx, lambda).into(),
            MaterialEnum::Dielectric(m) => m.bxdf(tex_eval, ctx, lambda).into(),
        }
    }
}

pub trait Material {
    type BxDF: crate::reflection::BxDF;

    fn bxdf(
        &self,
        tex_eval: &impl TextureEvaluator,
        ctx: &MaterialEvalContext,
        lambda: &mut SampledWavelengths,
    ) -> Self::BxDF;

    fn bsdf<'a>(
        &self,
        tex_eval: &impl TextureEvaluator,
        ctx: &MaterialEvalContext,
        lambda: &mut SampledWavelengths,
        scratch_buffer: &'a mut ScratchBuffer,
    ) -> BSDF<'a, Self::BxDF> {
        let bxdf = scratch_buffer.alloc(self.bxdf(tex_eval, ctx, lambda));
        BSDF::new(ctx.ns, ctx.dpdus, bxdf)
    }
}

#[derive(Clone, Debug)]
pub struct MaterialEvalContext {
    pub tex_eval_ctx: TextureEvalContext,
    pub wo: Vec3f,
    pub ns: Normal3f,
    pub dpdus: Vec3f,
}

impl MaterialEvalContext {
    pub fn from_surface_interaction(si: &SurfaceInteraction) -> Self {
        Self {
            tex_eval_ctx: TextureEvalContext::from_surface_interaction(si),
            wo: si.wo.unwrap(),
            ns: si.shading.n,
            dpdus: si.shading.dpdu,
        }
    }
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

#[derive(Default)]
pub struct UniversalTextureEvaluator {}

impl UniversalTextureEvaluator {
    pub fn new() -> Self {
        Self::default()
    }
}

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

#[derive(Debug)]
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
        lambda: &mut SampledWavelengths,
    ) -> Self::BxDF {
        let reflectance = tex_eval
            .eval_spectrum(&*self.reflectance, &ctx.tex_eval_ctx, lambda)
            .clamp(0.0, 1.0);
        DiffuseBxDF::new(reflectance)
    }
}

impl From<DiffuseMaterial> for MaterialEnum {
    fn from(value: DiffuseMaterial) -> Self {
        Self::Diffuse(value)
    }
}

#[derive(Debug)]
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
        lambda: &mut SampledWavelengths,
    ) -> Self::BxDF {
        // Compute index of refraction
        let sampled_eta = self.eta.at(lambda[0]);
        if !self.eta.is_constant() {
            lambda.terminate_secondary();
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
