use enum_dispatch::enum_dispatch;

use crate::core::{
    constants::{INV_4_PI, PI},
    safe_sqrt, spherical_direction, Float, Frame, Point2f, Vec3f,
};

#[enum_dispatch]
#[derive(Clone, Debug, PartialEq)]
pub enum PhaseFunctionEnum {
    HenyeyGreenstein,
}

#[enum_dispatch(PhaseFunctionEnum)]
pub trait PhaseFunction {
    fn value(&self, outgoing_dir: Vec3f, incident_dir: Vec3f) -> Float;
    fn sample(&self, outgoing_dir: Vec3f, u: Point2f) -> Option<PhaseFunctionSample>;
    fn pdf(&self, outgoing_dir: Vec3f, incident_dir: Vec3f) -> Float;
}

pub struct PhaseFunctionSample {
    pub value: Float,
    pub incident_dir: Vec3f,
    pub pdf: Float,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HenyeyGreenstein {
    g: Float,
}

impl HenyeyGreenstein {
    pub fn new(g: Float) -> Self {
        Self { g }
    }
}

impl PhaseFunction for HenyeyGreenstein {
    fn value(&self, outgoing_dir: Vec3f, incident_dir: Vec3f) -> Float {
        henyey_greenstein(outgoing_dir.dot(incident_dir), self.g)
    }

    fn sample(&self, outgoing_dir: Vec3f, u: Point2f) -> Option<PhaseFunctionSample> {
        let (incident_dir, pdf) = sample_henyey_greenstein(outgoing_dir, self.g, u);
        Some(PhaseFunctionSample {
            value: pdf,
            incident_dir,
            pdf,
        })
    }

    fn pdf(&self, outgoing_dir: Vec3f, incident_dir: Vec3f) -> Float {
        self.value(outgoing_dir, incident_dir)
    }
}

fn henyey_greenstein(cos_theta: Float, g: Float) -> Float {
    let denom = 1.0 + g * g + 2.0 * g * cos_theta;
    INV_4_PI * (1.0 - g * g) / (denom * safe_sqrt(denom))
}

fn sample_henyey_greenstein(out_dir: Vec3f, g: Float, u: Point2f) -> (Vec3f, Float) {
    // Compute cos theta
    let cos_theta = if g.abs() < 1e-3 {
        1.0 - 2.0 * u[0]
    } else {
        -1.0 / (2.0 * g) * (1.0 + g * g - ((1.0 - g * g) / (1.0 + g - 2.0 * g * u[0])).powi(2))
    };

    // Compute outgoing dir
    let sin_theta = safe_sqrt(1.0 - cos_theta * cos_theta);
    let phi = 2.0 * PI * u[1];
    let w_frame = Frame::from_z(out_dir);
    let in_dir = w_frame.from_local(spherical_direction(sin_theta, cos_theta, phi));

    // Compute PDF (equal to value, because HG func is normalized)
    let pdf = henyey_greenstein(cos_theta, g);

    (in_dir, pdf)
}
