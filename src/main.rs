use pbrt_rs::{
    geometry::Transform,
    math::{Point3f, Vec3f},
    sampling::IndependentSampler,
    shapes::Sphere,
};

fn main() {}

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
}
