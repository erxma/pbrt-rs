use std::sync::Arc;

use pbrt_rs::{
    camera::{CameraTransform, Film, PerspectiveCamera, PixelSensor, RGBFilm},
    color::{RGB, SRGB},
    geometry::{Bounds2f, Bounds2i, Transform},
    image::BoxFilter,
    lights::{DirectionalLight, UniformInfiniteLight},
    math::{Point2f, Point2i, Point3f, Vec2f, Vec3f},
    sampling::{
        spectrum::{BlackbodySpectrum, RgbAlbedoSpectrum},
        IndependentSampler,
    },
    shapes::{BilinearPatch, BilinearPatchMesh, Sphere},
    Float,
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

    let filter = Arc::new(BoxFilter::new(Vec2f::new(0.5, 0.5)).into());

    let sensor = Arc::new(
        PixelSensor::builder()
            .output_color_space(&SRGB)
            .imaging_ratio(1.0)
            .build()
            .unwrap(),
    );

    let film = Arc::new(
        RGBFilm::builder()
            .full_resolution(Point2i::new(400, 400))
            .pixel_bounds(Bounds2i::new(Point2i::new(0, 0), Point2i::new(400, 400)))
            .diagonal(35.0)
            .filter(filter)
            .sensor(sensor)
            .color_space(&SRGB)
            .max_component_value(Float::INFINITY)
            .build()
            .unwrap()
            .into(),
    );

    let camera = PerspectiveCamera::builder()
        .transform(CameraTransform::new(look_at))
        .shutter_open(0.0)
        .shutter_close(1.0)
        .film(film)
        .fov(45.0)
        .screen_window(Bounds2f::new(
            Point2f::new(-1.0, -1.0),
            Point2f::new(1.0, 1.0),
        ))
        .lens_radius(0.0)
        .focal_distance(10e30)
        .build()
        .unwrap();

    let inf_light = UniformInfiniteLight::new(
        Transform::IDENTITY,
        &RgbAlbedoSpectrum::new(&SRGB, RGB::new(0.4, 0.45, 0.5)),
        1.0,
    );

    let sun_light = DirectionalLight::new(
        Transform::look_at(Point3f::new(-30.0, 40.0, 100.0), Point3f::ZERO, Vec3f::UP),
        &BlackbodySpectrum::new(3000.0),
        1.5,
    );

    const VERTS: [usize; 4] = [0, 1, 2, 3];
    const POS: [Point3f; 4] = [
        Point3f::new(-20.0, -20.0, 0.0),
        Point3f::new(20.0, -20.0, 0.0),
        Point3f::new(-20.0, 20.0, 0.0),
        Point3f::new(20.0, 20.0, 0.0),
    ];
    const UV: [Point2f; 4] = [
        Point2f::new(0.0, 1.0),
        Point2f::new(1.0, 0.0),
        Point2f::new(1.0, 1.0),
        Point2f::new(0.0, 1.0),
    ];
    BilinearPatchMesh::init_mesh_data(vec![BilinearPatchMesh {
        vertices: &VERTS,
        positions: &POS,
        normals: None,
        uv: Some(&UV),
        reverse_orientation: false,
        transform_swaps_handedness: false,
        image_distribution: None,
    }]);
    let patch = BilinearPatch::new(BilinearPatchMesh::get(0).unwrap(), 0, 0);
}
