use std::path::PathBuf;

use crate::{
    core::Float,
    scene_parsing::{
        common::{impl_from_entity, EntityDirective, FromEntity},
        PbrtParseError,
    },
};

#[derive(Clone, Debug, PartialEq)]
pub enum Film {
    Rgb(RgbFilm),
}

impl Film {
    pub fn aspect_ratio(&self) -> Float {
        match self {
            Film::Rgb(film) => film.x_resolution as Float / film.y_resolution as Float,
        }
    }
}

impl Default for Film {
    fn default() -> Self {
        Self::Rgb(RgbFilm::default())
    }
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
    pub x_resolution: usize,
    pub y_resolution: usize,
    pub crop_window: [Float; 4],
    //pub pixel_bounds: [usize; 4],
    pub diagonal: Float,
    pub filename: PathBuf,
    pub save_fp16: bool,
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
