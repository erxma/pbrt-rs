use crate::{
    geometry::{
        bounds3::Bounds3f,
        interaction::{Interaction, SurfaceInteraction},
        ray::Ray,
    },
    math::{
        normal3::Normal3f,
        point::{Point2f, Point3f, Point3fi},
        vec::Vec3f,
    },
    sampling::spectrum::{SampledSpectrum, SampledWavelengths},
    Float,
};

pub trait Light {
    /// The total emitted power (Φ) of this light, at the given `wavelengths`.
    fn phi(&self, wavelengths: &SampledWavelengths) -> SampledSpectrum;

    /// This light's light category.
    fn light_type(&self) -> LightType;

    /// Returns `true` if `self` is defined using a Dirac delta distribution,
    /// `false` otherwise.
    fn is_delta_light(&self) -> bool {
        match self.light_type() {
            LightType::DeltaPosition | LightType::DeltaDirection => true,
            LightType::Area | LightType::Infinite => false,
        }
    }

    /// Given `ctx` info about a point in the scene,
    /// returns info about the incident radiance and PDF for the point,
    /// for this light.
    ///
    /// If no light can reach the point, or there is no valid sample associated
    /// with `u`, returns `None`.
    ///
    /// `allow_incomplete_pdf`: Wheter to allow skipping sample for directions
    /// where the constribution is small.
    fn sample_li(
        ctx: LightSampleContext,
        u: Point2f,
        wavelengths: &SampledWavelengths,
        allow_incomplete_pdf: bool,
    ) -> Option<LightLiSample>;

    /// Returns the PDF for sampling the direction `wi` from the point in `ctx`.
    ///
    /// Assumes that such a ray has already been found.
    fn pdf_li(&self, ctx: LightSampleContext, wi: Vec3f, allow_incomplete_pdf: bool) -> Float;

    /// Finds the radiance emitted back along a ray,
    /// given local info about the intersection ppoint and outgoing direction.
    ///
    /// Should not be called for any light with no associated geometry.
    fn radiance(
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
    fn radiance_infinite(&self, ray: &Ray, wavelengths: &SampledWavelengths) -> SampledSpectrum;

    /// Preprocessing step to invoked before rendering.
    fn preprocess(&self, scene_bounds: Bounds3f);
}

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
