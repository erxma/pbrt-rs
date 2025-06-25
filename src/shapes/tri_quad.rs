use std::{fs::File, path::Path};

use itertools::Itertools;
use log::warn;
use num_traits::AsPrimitive;
use ply_rs_bw as ply;
use ply_rs_bw::ply::{DefaultElement, KeyMap, Ply, Property};
use thiserror::Error;

use crate::core::{Float, Normal3f, Point2f, Point3f};

#[derive(Debug)]
pub struct TriQuadMesh {
    /// Index array for the triangle faces in the mesh.
    pub tri_indices: Vec<usize>,
    /// Index array for the quad faces in the mesh.
    pub quad_indices: Vec<usize>,
    /// Vertex positions in render space.
    pub positions: Vec<Point3f>,
    /// Per-vertex normals in render space, if any.
    pub normals: Option<Vec<Normal3f>>,
    /// Vertex UVs, if any.
    pub uv: Option<Vec<Point2f>>,
}

impl TriQuadMesh {
    /// Load a mesh from a `.ply` file.
    pub fn load_file(path: impl AsRef<Path>) -> Result<Self, FromPlyError> {
        let mut file = File::open(path)?;
        let parser = ply::parser::Parser::<DefaultElement>::new();
        let ply = parser.read_ply(&mut file)?;

        Self::try_from(ply)
    }
}

#[derive(Error, Debug)]
pub enum FromPlyError {
    #[error("io error - {0}")]
    Io(#[from] std::io::Error),
    #[error("expected element {0} in file")]
    MissingElement(&'static str),
    #[error("expected property {key} on {element}")]
    MissingProperty { key: String, element: &'static str },
    #[error("unexpected property type for `{key}`, expected {expected}")]
    InvalidPropertyType { key: String, expected: &'static str },
}

impl TryFrom<Ply<DefaultElement>> for TriQuadMesh {
    type Error = FromPlyError;

    fn try_from(mut ply: Ply<DefaultElement>) -> Result<Self, Self::Error> {
        // Get the vertex and face elements
        let vertex_element = ply
            .header
            .elements
            .get("vertex")
            .ok_or(FromPlyError::MissingElement("vertex"))?;
        let face_element = ply
            .header
            .elements
            .get("face")
            .ok_or(FromPlyError::MissingElement("face"))?;

        // Extract vertex and face count
        let n_vertices = vertex_element.count;
        let n_faces = face_element.count;

        // Check if normals and UVs are expected based on presence of one
        // of the keys for them
        let has_normals = vertex_element.properties.contains_key("nx");
        let has_uvs = vertex_element.properties.contains_key("u")
            || vertex_element.properties.contains_key("texture_u")
            || vertex_element.properties.contains_key("s")
            || vertex_element.properties.contains_key("texture_s");

        // Allocate vec for all the data
        let mut positions = Vec::with_capacity(n_vertices);
        // If normals or UVs aren't expected, just allocate empty
        let mut normals = Vec::with_capacity(if has_normals { n_vertices } else { 0 });
        let mut uvs = Vec::with_capacity(if has_uvs { n_vertices } else { 0 });
        let mut tri_indices = Vec::with_capacity(n_faces * 3);
        let mut quad_indices = Vec::with_capacity(n_faces * 4);

        // Helper to try to get a property with a float-like type
        // and one of the options for key name
        fn take_float_prop(
            element_name: &'static str,
            element: &mut KeyMap<Property>,
            key_options: &[&'static str],
        ) -> Result<Float, FromPlyError> {
            let mut prop = None;
            for &key in key_options {
                let val = element.remove(key);
                if val.is_some() {
                    prop = val;
                    break;
                }
            }
            let prop = prop.ok_or_else(|| FromPlyError::MissingProperty {
                key: key_options.join("/"),
                element: element_name,
            })?;

            let val = match prop {
                Property::Float(val) => val as Float,
                Property::Double(val) => val as Float,
                _ => {
                    return Err(FromPlyError::InvalidPropertyType {
                        key: key_options.join("/"),
                        expected: "float or double",
                    })
                }
            };
            Ok(val)
        }

        // Flag to prevent logging warning for extra props for every vertex
        let mut warned_vertex_extra_props = false;
        // For each vertex in the payload:
        for vertex in ply.payload.remove("vertex").unwrap().iter_mut() {
            // Extract (remove) props x, y, z for positions
            let x = take_float_prop("vertex", vertex, &["x"])?;
            let y = take_float_prop("vertex", vertex, &["y"])?;
            let z = take_float_prop("vertex", vertex, &["z"])?;
            positions.push(Point3f::new(x, y, z));

            // If normals are expected, extract nx, ny, nz
            if has_normals {
                let nx = take_float_prop("vertex", vertex, &["nx"])?;
                let ny = take_float_prop("vertex", vertex, &["ny"])?;
                let nz = take_float_prop("vertex", vertex, &["nz"])?;
                normals.push(Normal3f::new(nx, ny, nz));
            }

            // If UVs are expected, extract u, v (multiple aliases supported)
            if has_uvs {
                let u = take_float_prop("vertex", vertex, &["u", "texture_u", "s", "texture_s"])?;
                let v = take_float_prop("vertex", vertex, &["v", "texture_v", "t", "texture_t"])?;
                uvs.push(Point2f::new(u, v));
            }

            // Log a warning if vertex has remaining unextracted props
            // and haven't done this already
            if !warned_vertex_extra_props && !vertex.is_empty() {
                warn!(
                    "Skipping unrecognized properties '{}' on vertex",
                    vertex.keys().join(", ")
                );
                warned_vertex_extra_props = true;
            }
        }

        // Helper to `as` convert index type into usize
        fn convert_face_indices<T: AsPrimitive<usize>>(indices: Vec<T>) -> Vec<usize> {
            indices.into_iter().map(T::as_).collect()
        }

        // Flag to prevent logging warning for extra props for every face
        let mut warned_face_extra_props = false;
        // For each face in the payload:
        for (i, face) in ply.payload.remove("face").unwrap().iter_mut().enumerate() {
            let face_err = FromPlyError::MissingProperty {
                key: "vertex_indices".into(),
                element: "face",
            };

            // Extract the vertex_indices list and convert type to usize's if compatible
            let mut indices: Vec<usize> = match face.remove("vertex_indices").ok_or(face_err)? {
                Property::ListShort(items) => convert_face_indices(items),
                Property::ListUShort(items) => convert_face_indices(items),
                Property::ListInt(items) => convert_face_indices(items),
                Property::ListUInt(items) => convert_face_indices(items),
                _ => {
                    return Err(FromPlyError::InvalidPropertyType {
                        key: "vertex_indices".to_owned(),
                        expected: "list of short, ushort, int, or uint",
                    });
                }
            };

            // Depending on index count, append to triangle or quad index list,
            // or ignore and warn on any others
            match indices.len() {
                3 => tri_indices.append(&mut indices),
                4 => quad_indices
                    .extend_from_slice(&[indices[0], indices[1], indices[3], indices[2]]),
                n => warn!(
                    "Ignoring plymesh face with {n} indices, only triangles and quads expected"
                ),
            }

            // Log a warning if face has remaining unextracted props
            // and haven't done this already
            if !warned_face_extra_props && !face.is_empty() {
                warn!(
                    "Skipping unrecognized properties '{}' on face {i}",
                    face.keys().join(", ")
                );
                warned_face_extra_props = true;
            }
        }

        // Put everything together
        Ok(Self {
            tri_indices,
            quad_indices,
            positions,
            // Only use the vecs for normals and uvs if expecting to, put None otherwise
            normals: if has_normals { Some(normals) } else { None },
            uv: if has_uvs { Some(uvs) } else { None },
        })
    }
}
