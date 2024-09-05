use std::{path::PathBuf, sync::Arc};

use log::info;
use pbrt_rs::{
    camera::{Camera, CameraTransform, PerspectiveCamera, PixelSensor, RGBFilm},
    color::{RGB, SRGB},
    geometry::{Bounds2f, Bounds2i, Transform},
    image::BoxFilter,
    integrators::{Integrate, SimplePathIntegrator},
    lights::{DirectionalLight, UniformInfiniteLight},
    materials::{
        ConstantFloatTexture, ConstantSpectrumTexture, DielectricMaterial, DiffuseMaterial,
        FloatTextureEnum,
    },
    math::{Point2f, Point2i, Point3f, Vec2f, Vec3f},
    primitives::{BVHAggregate, BVHSplitMethod, SimplePrimitive},
    sampling::{
        spectrum::{
            BlackbodySpectrum, ConstantSpectrum, RgbAlbedoSpectrum, RgbIlluminantSpectrum, Spectrum,
        },
        IndependentSampler,
    },
    shapes::{BilinearPatch, BilinearPatchMesh, Sphere},
    Float,
};
use time::{macros::format_description, OffsetDateTime};

fn main() {
    let log_env = env_logger::Env::default().default_filter_or("info");
    env_logger::init_from_env(log_env);
    info!("Initialized logger.");

    render_cpu();
}

fn render_cpu() {
    let world_to_camera = Transform::look_at(
        Point3f::new(3.0, 4.0, 1.5),
        Point3f::new(0.5, 0.5, 0.0),
        Vec3f::new(0.0, 0.0, 1.0),
    );

    let sampler = IndependentSampler::new(256, None);

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
            .filename(PathBuf::from(format!(
                "render_{}.exr",
                OffsetDateTime::now_local()
                    .unwrap()
                    .format(&format_description!(
                        "[year]-[month]-[day]T[hour]:[minute]:[second]"
                    ))
                    .unwrap()
            )))
            .color_space(&SRGB)
            .max_component_value(Float::INFINITY)
            .build()
            .unwrap()
            .into(),
    );

    let camera = PerspectiveCamera::builder()
        .transform(CameraTransform::new(world_to_camera.inverse()))
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

    let inf_spec = RgbIlluminantSpectrum::new(&SRGB, RGB::new(0.4, 0.45, 0.5));
    let inf_light =
        Arc::new(UniformInfiniteLight::new(&inf_spec, 1.0 / inf_spec.to_photometric()).into());

    let sun_from = Point3f::new(-30.0, 40.0, 100.0);
    let sun_to = Point3f::new(0.0, 0.0, 1.0);
    let w = (sun_from - sun_to).normalized();
    let (w, v1, v2) = w.coordinate_system();
    let sun_transform = Transform::from_arr([
        [v1.x(), v2.x(), w.x(), 0.0],
        [v1.y(), v2.y(), w.y(), 0.0],
        [v1.z(), v2.z(), w.z(), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]);

    let sun_spec = BlackbodySpectrum::new(3000.0);
    let sun_light = Arc::new(
        DirectionalLight::new(
            camera.camera_transform().render_from_world(sun_transform),
            &sun_spec,
            1.5 / sun_spec.to_photometric(),
        )
        .into(),
    );

    let sphere = Arc::new(
        Sphere::builder()
            .radius(1.0)
            .z_min(-1.0)
            .z_max(1.0)
            .phi_max(360.0)
            .render_from_object(
                camera
                    .camera_transform()
                    .render_from_world(Transform::IDENTITY),
            )
            .reverse_orientation(false)
            .build()
            .unwrap()
            .into(),
    );
    let rough_tex: Arc<FloatTextureEnum> = Arc::new(ConstantFloatTexture::new(0.0).into());
    let eta = Arc::new(ConstantSpectrum::new(1.5).into());
    let sphere_mat =
        Arc::new(DielectricMaterial::new(rough_tex.clone(), rough_tex, false, eta).into());
    let sphere_prim = Arc::new(SimplePrimitive::new(sphere, sphere_mat).into());

    let floor_indices = vec![0, 1, 2, 3];
    let floor_pos = vec![
        Point3f::new(-20.0, -20.0, 0.0),
        Point3f::new(20.0, -20.0, 0.0),
        Point3f::new(-20.0, 20.0, 0.0),
        Point3f::new(20.0, 20.0, 0.0),
    ];
    let floor_uv = vec![
        Point2f::new(0.0, 1.0),
        Point2f::new(1.0, 0.0),
        Point2f::new(1.0, 1.0),
        Point2f::new(0.0, 1.0),
    ];
    BilinearPatchMesh::init_mesh_data(vec![BilinearPatchMesh::new(
        &camera
            .camera_transform()
            .render_from_world(Transform::translate(Vec3f::new(0.0, 0.0, -1.0))),
        false,
        floor_indices,
        floor_pos,
        None,
        Some(floor_uv),
    )]);
    let floor_patch = Arc::new(BilinearPatch::new(BilinearPatchMesh::get(0).unwrap(), 0, 0).into());
    let floor_mat_tex = Arc::new(
        ConstantSpectrumTexture::new(RgbAlbedoSpectrum::new(&SRGB, RGB::new(0.8, 0.8, 0.8)).into())
            .into(),
    );
    let floor_mat = Arc::new(DiffuseMaterial::new(floor_mat_tex).into());
    let floor_prim = Arc::new(SimplePrimitive::new(floor_patch, floor_mat).into());

    let aggregate = BVHAggregate::new(vec![sphere_prim, floor_prim], 255, BVHSplitMethod::Middle);
    let lights = vec![inf_light, sun_light];
    let mut integrator = SimplePathIntegrator::new(
        5,
        true,
        true,
        camera.into(),
        sampler.into(),
        aggregate.into(),
        lights,
    );

    integrator.render();
}
