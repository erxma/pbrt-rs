use std::path::PathBuf;

use crate::{
    core::common::*,
    scene_parsing::common::{
        params_map_to_fields, Alpha, EntityDirective, FromEntity, GraphicsState, PbrtParseError,
        Value,
    },
};

#[derive(Clone, Debug, PartialEq)]
pub enum ShapeDesc {
    Sphere(Sphere),
    BilinearMesh(BilinearMesh),
    TriangleMesh(TriangleMesh),
    PlyMesh(PlyMesh),
}

impl ShapeDesc {
    pub fn material_name(&self) -> &str {
        match self {
            ShapeDesc::Sphere(desc) => &desc.material_name,
            ShapeDesc::BilinearMesh(desc) => &desc.material_name,
            ShapeDesc::TriangleMesh(desc) => &desc.material_name,
            ShapeDesc::PlyMesh(desc) => &desc.material_name,
        }
    }
}

impl FromEntity for ShapeDesc {
    fn from_entity(entity: EntityDirective, state: &GraphicsState) -> Result<Self, PbrtParseError> {
        assert_eq!(entity.identifier, "Shape");

        match entity.subtype {
            "sphere" => Sphere::from_entity(entity, state).map(ShapeDesc::Sphere),
            "bilinearmesh" => BilinearMesh::from_entity(entity, state).map(ShapeDesc::BilinearMesh),
            "trianglemesh" => TriangleMesh::from_entity(entity, state).map(ShapeDesc::TriangleMesh),
            "plymesh" => PlyMesh::from_entity(entity, state).map(ShapeDesc::PlyMesh),
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
    pub world_from_object: Transform,
    pub reverse_orientation: bool,
    pub material_name: String,

    pub radius: Float,
    pub z_min: Float,
    pub z_max: Float,
    pub phi_max: Float,
}

impl Default for Sphere {
    fn default() -> Self {
        Self {
            alpha: Alpha::Constant(1.0),
            world_from_object: Transform::IDENTITY,
            reverse_orientation: false,
            material_name: Default::default(),

            radius: 1.0,
            z_min: -1.0,
            z_max: 1.0,
            phi_max: 360.0,
        }
    }
}

impl FromEntity for Sphere {
    fn from_entity(
        mut entity: EntityDirective,
        state: &GraphicsState,
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
        result.z_max = entity
            .param_map
            .remove("zmax")
            .unwrap_or(Value::Float(result.radius))
            .try_into()?;

        result.world_from_object = state.current_transform.clone();
        result.reverse_orientation = state.reverse_orientation;
        result.material_name =
            state
                .current_material_name
                .clone()
                .ok_or(PbrtParseError::MissingRequiredParameter(
                    "Material".to_string(),
                ))?;

        entity.param_map.check_no_remaining_params()?;

        Ok(result)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BilinearMesh {
    pub alpha: Alpha,
    pub world_from_object: Transform,
    pub reverse_orientation: bool,
    pub material_name: String,

    pub indices: Vec<usize>,
    pub positions: Vec<Point3f>,
    pub normals: Option<Vec<Normal3f>>,
    pub uvs: Option<Vec<Point2f>>,
}

impl Default for BilinearMesh {
    fn default() -> Self {
        Self {
            alpha: Alpha::Constant(1.0),
            world_from_object: Transform::default(),
            reverse_orientation: false,
            material_name: Default::default(),

            indices: vec![0, 1, 2, 3],
            positions: vec![],
            normals: None,
            uvs: None,
        }
    }
}

impl FromEntity for BilinearMesh {
    fn from_entity(
        mut entity: EntityDirective,
        state: &GraphicsState,
    ) -> Result<Self, PbrtParseError> {
        let mut result = Self::default();

        params_map_to_fields! {
            entity.param_map => result,
            required {
                positions = "P"
            }
            has_defaults {
                normals = "N",
                uvs = "uv"
            }
        }

        // Set indices array:
        // If missing, can default to [0, 1, 2, 3] if there are 4 vertices.
        // Otherwise, it must be specified.
        match entity.param_map.remove("indices") {
            Some(indices) => result.indices = indices.try_into()?,
            None if result.positions.len() == 4 => {}
            None => {
                return Err(PbrtParseError::MissingRequiredParameter(
                    "indices".to_string(),
                ))
            }
        }

        result.world_from_object = state.current_transform.clone();
        result.reverse_orientation = state.reverse_orientation;
        result.material_name =
            state
                .current_material_name
                .clone()
                .ok_or(PbrtParseError::MissingRequiredParameter(
                    "Material".to_string(),
                ))?;

        entity.param_map.check_no_remaining_params()?;

        Ok(result)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TriangleMesh {
    pub alpha: Alpha,
    pub world_from_object: Transform,
    pub reverse_orientation: bool,
    pub material_name: String,

    pub indices: Vec<usize>,
    pub positions: Vec<Point3f>,
    pub normals: Option<Vec<Normal3f>>,
    pub tangents: Option<Vec<Vec3f>>,
    pub uvs: Option<Vec<Point2f>>,
}

impl Default for TriangleMesh {
    fn default() -> Self {
        Self {
            alpha: Alpha::Constant(1.0),
            world_from_object: Transform::default(),
            reverse_orientation: false,
            material_name: Default::default(),

            indices: vec![0, 1, 2],
            positions: vec![],
            normals: None,
            tangents: None,
            uvs: None,
        }
    }
}

impl FromEntity for TriangleMesh {
    fn from_entity(
        mut entity: EntityDirective,
        state: &GraphicsState,
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

        // Set indices array:
        // If missing, can default to [0, 1, 2] if there are 3 vertices.
        // Otherwise, it must be specified.
        match entity.param_map.remove("indices") {
            Some(indices) => result.indices = indices.try_into()?,
            None if result.positions.len() == 3 => {}
            None => {
                return Err(PbrtParseError::MissingRequiredParameter(
                    "indices".to_string(),
                ))
            }
        }

        result.world_from_object = state.current_transform.clone();
        result.reverse_orientation = state.reverse_orientation;
        result.material_name =
            state
                .current_material_name
                .clone()
                .ok_or(PbrtParseError::MissingRequiredParameter(
                    "Material".to_string(),
                ))?;

        entity.param_map.check_no_remaining_params()?;

        Ok(result)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlyMesh {
    pub alpha: Alpha,
    pub world_from_object: Transform,
    pub reverse_orientation: bool,
    pub material_name: String,

    pub filename: PathBuf,
    pub displacement_name: Option<String>,
    pub edge_length: Float,
}

impl Default for PlyMesh {
    fn default() -> Self {
        Self {
            alpha: Alpha::Constant(1.0),
            world_from_object: Transform::default(),
            reverse_orientation: false,
            material_name: Default::default(),

            filename: PathBuf::new(),
            displacement_name: None,
            edge_length: 1.0,
        }
    }
}

impl FromEntity for PlyMesh {
    fn from_entity(
        mut entity: EntityDirective,
        state: &GraphicsState,
    ) -> Result<Self, PbrtParseError> {
        let mut result = Self::default();

        params_map_to_fields! {
            entity.param_map => result,
            required {
                filename = "filename"
            }
            has_defaults {
                displacement_name = "displacement",
                edge_length = "edgelength"
            }
        }

        result.world_from_object = state.current_transform.clone();
        result.reverse_orientation = state.reverse_orientation;
        result.material_name =
            state
                .current_material_name
                .clone()
                .ok_or(PbrtParseError::MissingRequiredParameter(
                    "Material".to_string(),
                ))?;

        entity.param_map.check_no_remaining_params()?;

        Ok(result)
    }
}
