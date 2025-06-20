use std::path::PathBuf;

use strum::EnumString;
use time::{macros::format_description, OffsetDateTime};

use crate::{
    core::Float,
    scene_parsing::common::{
        params_map_to_fields, EntityDirective, FromEntity, GraphicsState, PbrtParseError, Value,
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
    fn from_entity(entity: EntityDirective, state: &GraphicsState) -> Result<Self, PbrtParseError> {
        assert_eq!(entity.identifier, "Film");

        match entity.subtype {
            "rgb" => RgbFilm::from_entity(entity, state).map(Film::Rgb),
            invalid_type => Err(PbrtParseError::UnrecognizedVariant {
                entity: "Film".to_string(),
                variant_name: invalid_type.to_owned(),
            }),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RgbFilm {
    pub x_resolution: usize,
    pub y_resolution: usize,
    pub crop_window: [Float; 4],
    pub pixel_bounds: [usize; 4],
    pub diagonal: Float,
    pub filename: PathBuf,
    pub save_fp16: bool,
    pub iso: Float,
    pub white_balance_temp: Option<Float>,
    pub sensor: SensorName,
    pub max_component_value: Float,
}

impl Default for RgbFilm {
    fn default() -> Self {
        let x_resolution = 1280;
        let y_resolution = 720;
        Self {
            x_resolution,
            y_resolution,
            crop_window: [0.0, 1.0, 0.0, 1.0],
            pixel_bounds: [0, 0, x_resolution, y_resolution],
            diagonal: 35.0,
            filename: default_filename(),
            save_fp16: true,
            iso: 100.0,
            white_balance_temp: None,
            sensor: SensorName::Cie1931,
            max_component_value: Float::INFINITY,
        }
    }
}

impl FromEntity for RgbFilm {
    fn from_entity(
        mut entity: EntityDirective,
        _state: &GraphicsState,
    ) -> Result<Self, PbrtParseError> {
        let mut result = RgbFilm::default();

        params_map_to_fields! {
            entity.param_map => result,
            has_defaults {
                x_resolution = "xresolution",
                y_resolution = "yresolution",
                crop_window = "cropwindow",
                diagonal = "diagonal",
                filename = "filename",
                save_fp16 = "savefp16",
                iso = "iso",
                white_balance_temp = "whitebalance",
                sensor = "sensor",
                max_component_value = "maxcomponentvalue"
            }
        }

        if let Some(pixel_bounds) = entity.param_map.remove("pixelbounds") {
            result.pixel_bounds = pixel_bounds.try_into()?;
        } else {
            result.pixel_bounds = [0, 0, result.x_resolution, result.y_resolution];
        }

        entity.param_map.check_no_remaining_params()?;

        Ok(result)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, EnumString, strum::Display)]
pub enum SensorName {
    #[strum(serialize = "cie1931")]
    Cie1931,
    #[strum(serialize = "canon_eos_100d")]
    CanonEos100d,
}

impl TryFrom<Value> for SensorName {
    type Error = PbrtParseError;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        let incorrect_type_err = PbrtParseError::IncorrectType {
            expected: "sensor_name".to_string(),
            found: value.clone(),
        };

        if let Value::String(string) = value {
            if let Ok(sensor) = string.parse() {
                return Ok(sensor);
            }
        }

        Err(incorrect_type_err)
    }
}

// FIXME: Will panic during tests due to lack of local offset
fn default_filename() -> PathBuf {
    // If unspecified, default out file to "render_{timestamp}.exr"
    let timestamp = OffsetDateTime::now_local()
        .unwrap()
        .format(&format_description!(
            "[year]-[month]-[day]T[hour]:[minute]:[second]"
        ))
        .unwrap();
    PathBuf::from(format!("render_{timestamp}.exr"))
}
