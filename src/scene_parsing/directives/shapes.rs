use crate::{
    core::{Float, Normal3f, Point2f, Point3f, Vec3f},
    scene_parsing::common::{
        params_map_to_fields, Alpha, EntityDirective, FromEntity, ParseContext, PbrtParseError,
        Value,
    },
};

#[derive(Clone, Debug, PartialEq)]
pub enum ShapeDesc {
    Sphere(Sphere),
    BilinearMesh(BilinearMesh),
}

impl FromEntity for ShapeDesc {
    fn from_entity(entity: EntityDirective, ctx: &ParseContext) -> Result<Self, PbrtParseError> {
        assert_eq!(entity.identifier, "Shape");

        match entity.subtype {
            "sphere" => Sphere::from_entity(entity, ctx).map(ShapeDesc::Sphere),
            "bilinearmesh" => BilinearMesh::from_entity(entity, ctx).map(ShapeDesc::BilinearMesh),
            invalid_type => Err(PbrtParseError::UnrecognizedVariant {
                entity: "Shape".to_string(),
                variant_name: invalid_type.to_owned(),
            }),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Sphere {
    pub alpha: Alpha,
    pub radius: Float,
    pub z_min: Float,
    pub z_max: Float,
    pub phi_max: Float,
}

impl Default for Sphere {
    fn default() -> Self {
        Self {
            alpha: Alpha::Constant(1.0),
            radius: 1.0,
            z_min: 1.0,
            z_max: 1.0,
            phi_max: 360.0,
        }
    }
}

impl FromEntity for Sphere {
    fn from_entity(
        mut entity: EntityDirective,
        _ctx: &ParseContext,
    ) -> Result<Self, PbrtParseError> {
        let mut result = Self::default();

        params_map_to_fields! {
            entity.param_map => result,
            has_defaults {
                alpha = "alpha",
                radius = "radius",
                phi_max = "phimax"
            }
        }

        // zmin, zmax default to -radius, radius from above
        result.z_min = entity
            .param_map
            .remove("zmin")
            .unwrap_or(Value::Float(-result.radius))
            .try_into()?;
        result.z_min = entity
            .param_map
            .remove("zmax")
            .unwrap_or(Value::Float(result.radius))
            .try_into()?;

        Ok(result)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BilinearMesh {
    pub alpha: Alpha,
    pub indices: Vec<usize>,
    pub positions: Vec<Point3f>,
    pub normals: Option<Vec<Normal3f>>,
    pub tangents: Option<Vec<Vec3f>>,
    pub uvs: Option<Vec<Point2f>>,
}

impl Default for BilinearMesh {
    fn default() -> Self {
        Self {
            alpha: Alpha::Constant(1.0),
            indices: vec![0, 1, 2],
            positions: vec![],
            normals: None,
            tangents: None,
            uvs: None,
        }
    }
}

impl FromEntity for BilinearMesh {
    fn from_entity(
        mut entity: EntityDirective,
        _ctx: &ParseContext,
    ) -> Result<Self, PbrtParseError> {
        let mut result = Self::default();

        params_map_to_fields! {
            entity.param_map => result,
            required {
                positions = "P"
            }
            has_defaults {
                normals = "N",
                tangents = "S",
                uvs = "uv"
            }
        }

        match entity.param_map.remove("indices") {
            Some(indices) => result.indices = indices.try_into()?,
            None if result.positions.len() == 3 => {}
            None => {
                return Err(PbrtParseError::MissingRequiredParameter(
                    "indices".to_string(),
                ))
            }
        }

        entity.param_map.check_no_remaining_params()?;

        Ok(result)
    }
}
