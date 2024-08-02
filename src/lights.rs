use crate::{
    geometry::{
        Bounds3f, Interaction, MediumInterfaceInteraction, Ray, SurfaceInteraction, Transform,
    },
    math::{Normal3f, Point2f, Point3f, Point3fi, Vec3f},
    medium::MediumInterface,
    memory::cache::{ArcIntern, ArcInternCache},
    sampling::spectrum::{
        DenselySampledSpectrum, SampledSpectrum, SampledWavelengths, Spectrum, SpectrumEnum,
    },
    Float, PI,
};
use delegate::delegate;
use derive_builder::Builder;
use enum_dispatch::enum_dispatch;
use std::sync::LazyLock;

#[enum_dispatch]
pub enum LightEnum<'a> {
    Point(PointLight<'a>),
}

impl<'a> LightEnum<'a> {
    delegate! {
        #[through(Light)]
        to self {
            /// The total emitted power (Φ) of this light, at the given `wavelengths`.
            pub fn phi(&self, wavelengths: &SampledWavelengths) -> SampledSpectrum;

            /// This light's light category.
            pub fn light_type(&self) -> LightType;

            /// Returns `true` if `self` is defined using a Dirac delta distribution,
            /// `false` otherwise.
            pub fn is_delta_light(&self) -> bool;

            /// Given `ctx` info about a point in the scene,
            /// returns info about the incident radiance and PDF for the point,
            /// for this light.
            ///
            /// If no light can reach the point, or there is no valid sample associated
            /// with `u`, returns `None`.
            ///
            /// `allow_incomplete_pdf`: Wheter to allow skipping sample for directions
            /// where the constribution is small.
            pub fn sample_li(
                &self,
                ctx: LightSampleContext,
                u: Point2f,
                wavelengths: &SampledWavelengths,
                allow_incomplete_pdf: bool,
            ) -> Option<LightLiSample>;

            /// Returns the PDF for sampling the direction `wi` from the point in `ctx`.
            ///
            /// Assumes that such a ray has already been found.
            pub fn pdf_li(&self, ctx: LightSampleContext, wi: Vec3f, allow_incomplete_pdf: bool) -> Float;

            /// Finds the radiance emitted back along a ray,
            /// given local info about the intersection ppoint and outgoing direction.
            ///
            /// Should not be called for any light with no associated geometry.
            pub fn radiance(
                &self,
                p: Point3f,
                n: Normal3f,
                uv: Point2f,
                w: Vec3f,
                wavelengths: &SampledWavelengths,
            ) -> SampledSpectrum;

            /// Finds the radiance contributed to ray by an infinite light.
            ///
            /// Should only be used for lights that are `LightType::Infinite`.
            pub fn radiance_infinite(&self, ray: &Ray, wavelengths: &SampledWavelengths) -> SampledSpectrum;

            /// Preprocessing step to invoked before rendering.
            pub fn preprocess(&self, scene_bounds: Bounds3f);
        }
    }
}

#[enum_dispatch(LightEnum)]
pub trait Light {
    fn phi(&self, wavelengths: &SampledWavelengths) -> SampledSpectrum;

    fn light_type(&self) -> LightType;

    fn is_delta_light(&self) -> bool {
        match self.light_type() {
            LightType::DeltaPosition | LightType::DeltaDirection => true,
            LightType::Area | LightType::Infinite => false,
        }
    }

    fn sample_li(
        &self,
        ctx: LightSampleContext,
        u: Point2f,
        wavelengths: &SampledWavelengths,
        allow_incomplete_pdf: bool,
    ) -> Option<LightLiSample>;

    fn pdf_li(&self, ctx: LightSampleContext, wi: Vec3f, allow_incomplete_pdf: bool) -> Float;

    fn radiance(
        &self,
        p: Point3f,
        n: Normal3f,
        uv: Point2f,
        w: Vec3f,
        wavelengths: &SampledWavelengths,
    ) -> SampledSpectrum;

    fn radiance_infinite(&self, ray: &Ray, wavelengths: &SampledWavelengths) -> SampledSpectrum;

    fn preprocess(&self, scene_bounds: Bounds3f);
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum LightType {
    /// Light that emits solely from a single point in space.
    DeltaPosition,
    /// Light that emits radiance along a single direction.
    DeltaDirection,
    /// Light that emits radiance from the surface of a geometric shape.
    Area,
    /// Light at "infinity" that does not have associated geometry,
    /// but provides radiance to rays that escape the scene.
    Infinite,
}

pub struct LightSampleContext {
    pub pi: Point3fi,
    pub n: Option<Normal3f>,
    pub n_shading: Option<Normal3f>,
}

impl LightSampleContext {
    pub fn with_surface_interaction(si: &SurfaceInteraction) -> Self {
        Self {
            pi: si.common.pi,
            n: Some(si.n),
            n_shading: Some(si.shading.n),
        }
    }

    pub fn with_medium_interaction(intr: &Interaction) -> Self {
        assert!(!matches!(intr, Interaction::Surface(_)), "For SurfaceInteractions, with_surface_interaction should be used instead. This one loses the normals info.");

        Self {
            pi: intr.pi(),
            n: None,
            n_shading: None,
        }
    }

    pub fn p(&self) -> Point3f {
        self.pi.midpoints_only()
    }
}

pub struct LightLiSample<'a> {
    /// The amount of radiance leaving the light toward the receiving point,
    /// not including the effect of extinction or occulusion.
    pub l: SampledSpectrum,
    /// The direction allow which light arrives at the point.
    pub wi: Vec3f,
    /// The PDF value for the light sample, measured with respect to
    /// the solid angle at the receiving point.
    pub pdf: Float,
    /// The point from which light is being emitted.
    pub p_light: Interaction<'a>,
}

struct SpectrumCache;

impl SpectrumCache {
    fn lookup_spectrum(s: &impl Spectrum) -> ArcIntern<DenselySampledSpectrum> {
        // One-time initialized cache
        static CACHE: LazyLock<ArcInternCache<DenselySampledSpectrum>> =
            LazyLock::new(ArcInternCache::new);

        // Return unique DenselySampled from cache
        CACHE.lookup(DenselySampledSpectrum::new(s, None, None))
    }
}

#[derive(Builder)]
pub struct PointLight<'a> {
    #[builder(default = "LightType::DeltaPosition", setter(skip))]
    light_type: LightType,
    render_from_light: Transform,
    medium_interface: MediumInterface<'a>,

    #[builder(field(
        ty = "Option<&'a SpectrumEnum<'a>>",
        build = "SpectrumCache::lookup_spectrum(self.i.unwrap())"
    ))]
    i: ArcIntern<DenselySampledSpectrum>,
    scale: Float,
}

impl<'a> PointLight<'a> {
    pub fn builder() -> PointLightBuilder<'a> {
        PointLightBuilder::default()
    }
}

impl<'a> Light for PointLight<'a> {
    fn phi(&self, wavelengths: &SampledWavelengths) -> SampledSpectrum {
        4.0 * PI * self.scale * self.i.sample(wavelengths)
    }

    fn light_type(&self) -> LightType {
        self.light_type
    }

    fn sample_li(
        &self,
        ctx: LightSampleContext,
        _u: Point2f,
        wavelengths: &SampledWavelengths,
        _allow_incomplete_pdf: bool,
    ) -> Option<LightLiSample> {
        let p = &self.render_from_light * Point3f::ZERO;
        let wi = (p - ctx.p()).normalized();
        let li = self.scale * self.i.sample(wavelengths) / p.distance_squared(ctx.p());

        Some(LightLiSample {
            l: li,
            wi,
            pdf: 1.0,
            p_light: Interaction::MediumInterface(
                MediumInterfaceInteraction::with_point_and_interface(p, self.medium_interface),
            ),
        })
    }

    fn pdf_li(&self, _ctx: LightSampleContext, _wi: Vec3f, _allow_incomplete_pdf: bool) -> Float {
        0.0
    }

    fn radiance(
        &self,
        _p: Point3f,
        _n: Normal3f,
        _uv: Point2f,
        _w: Vec3f,
        _wavelengths: &SampledWavelengths,
    ) -> SampledSpectrum {
        SampledSpectrum::with_single_value(0.0)
    }

    fn radiance_infinite(&self, _ray: &Ray, _wavelengths: &SampledWavelengths) -> SampledSpectrum {
        SampledSpectrum::with_single_value(0.0)
    }

    fn preprocess(&self, _scene_bounds: Bounds3f) {}
}
