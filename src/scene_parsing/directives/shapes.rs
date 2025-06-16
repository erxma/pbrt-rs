use crate::{
    core::{Float, Normal3f, Point2f, Point3f, Vec3f},
    scene_parsing::common::{
        impl_from_entity, params_map_to_fields, Alpha, EntityDirective, FromEntity, ParseContext,
        PbrtParseError,
    },
};

#[derive(Clone, Debug, PartialEq)]
pub enum Shape {
    Sphere(Sphere),
    BilinearMesh(BilinearMesh),
}

impl FromEntity for Shape {
    fn from_entity(entity: EntityDirective, ctx: &ParseContext) -> Result<Self, PbrtParseError> {
        assert_eq!(entity.identifier, "Shape");

        match entity.subtype {
            "sphere" => Sphere::from_entity(entity, ctx).map(Shape::Sphere),
            "bilinearmesh" => BilinearMesh::from_entity(entity, ctx).map(Shape::BilinearMesh),
            invalid_type => Err(PbrtParseError::UnrecognizedVariant {
                entity: "Shape".to_string(),
                variant_name: invalid_type.to_owned(),
            }),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Sphere {
    alpha: Alpha,
    radius: Float,
    z_min: Option<Float>,
    z_max: Option<Float>,
    phi_max: Float,
}

impl Default for Sphere {
    fn default() -> Self {
        Self {
            alpha: Alpha::Constant(1.0),
            radius: 1.0,
            z_min: None,
            z_max: None,
            phi_max: 360.0,
        }
    }
}

impl_from_entity! {
    Sphere,
    has_defaults {
        "alpha" => alpha,
        "radius" => radius,
        "zmin" => z_min,
        "zmax" => z_max,
        "phimax" => phi_max,
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BilinearMesh {
    alpha: Alpha,
    indices: Vec<usize>,
    positions: Vec<Point3f>,
    normals: Option<Vec<Normal3f>>,
    tangents: Option<Vec<Vec3f>>,
    uvs: Option<Vec<Point2f>>,
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
