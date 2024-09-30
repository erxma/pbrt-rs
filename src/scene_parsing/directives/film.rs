use std::path::PathBuf;

use crate::{
    core::Float,
    scene_parsing::{
        common::{impl_from_entity, param_map, EntityDirective, FromEntity},
        PbrtParseError,
    },
};

#[derive(Clone, Debug, PartialEq)]
pub enum Film {
    Rgb(RgbFilm),
}

impl FromEntity for Film {
    fn from_entity(
        entity: EntityDirective,
        ctx: &crate::scene_parsing::common::ParseContext,
    ) -> Result<Self, PbrtParseError> {
        assert_eq!(entity.identifier, "Film");

        match entity.subtype {
            "rgb" => RgbFilm::from_entity(entity, ctx).map(Film::Rgb),
            invalid_type => Err(PbrtParseError::UnrecognizedSubtype {
                entity: "Film".to_string(),
                type_name: invalid_type.to_owned(),
            }),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RgbFilm {
    x_resolution: usize,
    y_resolution: usize,
    crop_window: [Float; 4],
    //pixel_bounds: [usize; 4],
    diagonal: Float,
    filename: PathBuf,
    save_fp16: bool,
}

impl Default for RgbFilm {
    fn default() -> Self {
        let x_resolution = 1280;
        let y_resolution = 720;
        Self {
            x_resolution,
            y_resolution,
            crop_window: [0.0, 1.0, 0.0, 1.0],
            //pixel_bounds: [0, x_resolution, 0, y_resolution],
            diagonal: 35.0,
            filename: "pbrt.exr".into(),
            save_fp16: true,
        }
    }
}

impl_from_entity! {
    RgbFilm,
    has_defaults {
        "xresolution" => x_resolution,
        "yresolution" => y_resolution,
        "cropwindow" => crop_window,
        //"pixelbounds" => pixel_bounds,
        "diagonal" => diagonal,
        "filename" => filename,
        "savefp16" => save_fp16,
    }
}
