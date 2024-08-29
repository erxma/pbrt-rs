use pbrt_rs::{
    color::{RGBColorSpace, RGB},
    geometry::Transform,
    math::{Point2f, Point3f, Vec3f},
    sampling::{spectrum::PiecewiseLinearSpectrum, IndependentSampler},
    shapes::Sphere,
    util::data::{CIE_ILLUM_D6500, SRGB},
};

fn main() {
    render_cpu();
}

fn render_cpu() {
    let look_at = Transform::look_at(
        Point3f::new(3.0, 4.0, 1.5),
        Point3f::new(0.5, 0.5, 0.0),
        Vec3f::UP,
    );

    let sampler = IndependentSampler::new(128, None);

    let sphere = Sphere::builder()
        .radius(1.0)
        .z_min(-1.0)
        .z_max(1.0)
        .phi_max(360.0)
        .render_from_object(Transform::IDENTITY)
        .object_from_render(Transform::IDENTITY)
        .reverse_orientation(false);

    let illumd65 = PiecewiseLinearSpectrum::from_interleaved(&CIE_ILLUM_D6500, true).into();
    let color_space = RGBColorSpace::new(
        Point2f::new(0.64, 0.33),
        Point2f::new(0.3, 0.6),
        Point2f::new(0.15, 0.06),
        illumd65,
        SRGB.clone(),
    );

    println!("{:?}", color_space.to_xyz(RGB::new(0.5, 0.5, 0.5)));
}
