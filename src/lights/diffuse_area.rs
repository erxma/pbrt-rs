use std::{
    hash::{DefaultHasher, Hash},
    sync::Arc,
};

use bon::bon;
use ordered_float::NotNan;

use crate::{
    color::{RGBColorSpace, RGB},
    core::{constants::PI, Bounds3f, Float, Normal3f, Point2Usize, Point2f, Point3f, Ray, Vec3f},
    imaging::{Image, WrapMode},
    lights::{base::SpectrumCache, Light, LightLiSample, LightSampleContext, LightType},
    materials::{
        FloatTexture, FloatTextureEnum, TextureEvalContext, TextureEvaluator as _,
        UniversalTextureEvaluator,
    },
    media::MediumInterface,
    memory::ArcIntern,
    sampling::spectrum::{
        DenselySampledSpectrum, RgbIlluminantSpectrum, SampledSpectrum, SampledWavelengths,
        Spectrum, SpectrumEnum,
    },
    shapes::{Shape, ShapeEnum, ShapeSampleContext},
    util::routines::HasherFloat as _,
};

#[derive(Debug)]
pub struct DiffuseAreaLight {
    medium_interface: MediumInterface,
    light_type: LightType,

    shape: Arc<ShapeEnum>,
    alpha: Option<FloatTextureEnum>,
    /// Cached shape area
    area: Float,
    two_sided: bool,
    emission: Emission,
    scale: Float,
    image_color_space: &'static RGBColorSpace,
}

// Used for construction
#[derive(Debug)]
pub enum AreaLightEmission<'a> {
    Image(Image),
    Uniform(&'a SpectrumEnum),
}

// The stored type
#[derive(Debug)]
enum Emission {
    Image(Image),
    Uniform(ArcIntern<DenselySampledSpectrum>),
}

#[bon]
impl DiffuseAreaLight {
    #[builder]
    pub fn new(
        medium_interface: MediumInterface,
        emission: AreaLightEmission<'_>,
        scale: Float,
        shape: Arc<ShapeEnum>,
        mut alpha: Option<FloatTextureEnum>,
        image_color_space: &'static RGBColorSpace,
        two_sided: bool,
    ) -> Self {
        let area = shape.area();

        // Special case handling for area lights with constant zero-valued alpha
        // textures to allow invisible area lights:
        // Set the alpha texture to None so that as far as the DiffuseAreaLight is concerned,
        // there is no alpha texture and the light is fully emissive.
        // However, such lights will never be intersected by rays
        // (because their associated primitives still have the alpha texture),
        // so we mark them as DeltaPosition lights here so that MIS isn't used
        // for direct illumination. Thus, light sampling is the only strategy used
        // and we get an unbiased (if potentially high variance) estimate.
        let mut light_type = LightType::Area;
        if let Some(FloatTextureEnum::Constant(constant_alpha)) = &alpha {
            if constant_alpha.eval(&TextureEvalContext::default()) == 0.0 {
                light_type = LightType::DeltaPosition;
                alpha = None;
            }
        }

        let emission = match emission {
            AreaLightEmission::Image(image) => {
                // TODO: Validate image descs
                Emission::Image(image)
            }
            AreaLightEmission::Uniform(spectrum) => {
                Emission::Uniform(SpectrumCache::lookup_spectrum(spectrum))
            }
        };

        Self {
            medium_interface,
            light_type,
            shape,
            alpha,
            area,
            two_sided,
            emission,
            scale,
            image_color_space,
        }
    }

    fn alpha_masked(&self, p: Point3f, uv: Point2f) -> bool {
        if let Some(alpha) = &self.alpha {
            let a = UniversalTextureEvaluator::new().eval(
                alpha,
                &TextureEvalContext {
                    p,
                    uv,
                    ..Default::default()
                },
            );
            if a >= 1.0 {
                false
            } else if a <= 0.0 {
                true
            } else {
                let mut hasher = DefaultHasher::new();
                NotNan::new(p.x()).unwrap().hash(&mut hasher);
                NotNan::new(p.y()).unwrap().hash(&mut hasher);
                NotNan::new(p.z()).unwrap().hash(&mut hasher);
                hasher.finish_float() > a
            }
        } else {
            false
        }
    }
}

impl Light for DiffuseAreaLight {
    fn phi(&self, wavelengths: &SampledWavelengths) -> SampledSpectrum {
        let radiance = match &self.emission {
            Emission::Image(image) => {
                // Compute average light image emission
                let mut radiance = SampledSpectrum::with_single_value(0.0);
                for (x, y) in
                    itertools::iproduct!(0..image.resolution().x(), 0..image.resolution().y())
                {
                    let mut rgb = RGB::default();
                    for channel in 0..3 {
                        rgb[channel] = image.get_channel(
                            Point2Usize::new(x, y),
                            channel,
                            (WrapMode::Clamp, WrapMode::Clamp),
                        );
                    }
                    radiance +=
                        RgbIlluminantSpectrum::new(self.image_color_space, rgb).sample(wavelengths);
                }
                radiance
            }
            Emission::Uniform(spectrum) => spectrum.sample(wavelengths) * self.scale,
        };

        PI * if self.two_sided { 2.0 } else { 1.0 } * self.area * radiance
    }

    fn light_type(&self) -> LightType {
        self.light_type
    }

    fn sample_li(
        &self,
        ctx: LightSampleContext,
        u: Point2f,
        wavelengths: &SampledWavelengths,
        _allow_incomplete_pdf: bool,
    ) -> Option<LightLiSample> {
        // Sample point on shape for light
        let shape_ctx = ShapeSampleContext::new(ctx.pi, ctx.n, ctx.n_shading, 0.0);
        let mut shape_sample = self.shape.sample_with_context(&shape_ctx, u)?;
        if shape_sample.pdf == 0.0
            || (shape_sample.intr.pi.midpoints() - ctx.pi_mids()).length_squared() == 0.0
        {
            return None;
        }
        shape_sample.intr.medium_interface = Some(&self.medium_interface);

        let intr = &shape_sample.intr;
        let p_light = shape_sample.intr.pi.midpoints();

        // Check sampled point on shape against alpha texture, if present
        if self.alpha_masked(p_light, intr.uv) {
            return None;
        }

        // Return sample
        let incident = (p_light - ctx.pi_mids()).normalized();
        let emitted_radiance = self.radiance(p_light, intr.n, intr.uv, -incident, wavelengths);
        if emitted_radiance.is_all_zero() {
            return None;
        }

        Some(LightLiSample {
            l: emitted_radiance,
            wi: incident,
            pdf: shape_sample.pdf,
            p_light,
            medium_interface: Some(&self.medium_interface),
        })
    }

    fn pdf_li(&self, ctx: LightSampleContext, wi: Vec3f, _allow_incomplete_pdf: bool) -> Float {
        let shape_ctx = ShapeSampleContext::new(ctx.pi, ctx.n, ctx.n_shading, 0.0);
        self.shape.pdf_with_context(&shape_ctx, wi)
    }

    fn radiance(
        &self,
        p: Point3f,
        n: Normal3f,
        mut uv: Point2f,
        w: Vec3f,
        wavelengths: &SampledWavelengths,
    ) -> SampledSpectrum {
        // Check for zero emitted radiance from point on area light
        if (!self.two_sided && n.dot_v(w) < 0.0) || self.alpha_masked(p, uv) {
            SampledSpectrum::with_single_value(0.0)
        } else {
            match &self.emission {
                Emission::Image(image) => {
                    // Return emmision using image
                    uv[1] = 1.0 - uv[1];

                    let mut rgb = RGB::default();
                    for channel in 0..3 {
                        rgb[channel] =
                            image.bilerp_channel(uv, channel, (WrapMode::Clamp, WrapMode::Clamp));
                    }
                    let spec = RgbIlluminantSpectrum::new(self.image_color_space, rgb);
                    self.scale * spec.sample(wavelengths)
                }
                Emission::Uniform(spectrum) => self.scale * spectrum.sample(wavelengths),
            }
        }
    }

    fn radiance_infinite(&self, _ray: &Ray, _wavelengths: &SampledWavelengths) -> SampledSpectrum {
        SampledSpectrum::with_single_value(0.0)
    }

    fn preprocess(&self, _scene_bounds: Bounds3f) {
        // Nothing to do
    }
}
