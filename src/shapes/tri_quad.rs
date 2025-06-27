use std::collections::HashMap;
use std::{fs::File, path::Path};

use itertools::Itertools;
use log::warn;
use num_traits::AsPrimitive;
use ply_rs_bw as ply;
use ply_rs_bw::ply::{DefaultElement, KeyMap, Ply, Property};
use thiserror::Error;

use crate::core::{Float, Normal3f, Point2f, Point3f};
use crate::parallel::parallel_map;

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

    /// Convert all quads in this mesh (if any) into triangles,
    /// adding to the triangle index list.
    pub fn convert_to_only_triangles(&mut self) {
        // Reserve the needed space
        self.tri_indices.reserve(3 * self.quad_indices.len() / 2);

        // For each quad, split into two new triangles and add
        for q in self.quad_indices.chunks(4) {
            self.tri_indices
                .extend_from_slice(&[q[0], q[1], q[3], q[0], q[3], q[2]]);
        }

        // Empty all the quad indices
        self.quad_indices = Vec::new();
    }

    /// Derive normals for all vertices in this mesh by averaging the normals of
    /// neighboring faces, i.e. the cross products of each pair of edges the vertex
    /// is in.
    ///
    /// This mesh must contain only triangles. Any existing normals will be overwritten.
    pub fn compute_normals(&mut self) {
        assert!(self.quad_indices.is_empty());

        let mut normals = vec![Normal3f::new(0.0, 0.0, 0.0); self.positions.len()];
        for tri in self.tri_indices.chunks(3) {
            let v10 = self.positions[tri[1]] - self.positions[tri[0]];
            let v21 = self.positions[tri[2]] - self.positions[tri[1]];

            let mut vn = v10.cross(v21);
            if vn.length_squared() > 0.0 {
                vn = vn.normalized();
            }

            normals[tri[0]] += vn.into();
            normals[tri[1]] += vn.into();
            normals[tri[2]] += vn.into();
        }

        for n_tri in normals.iter_mut() {
            if n_tri.length_squared() > 0.0 {
                *n_tri = n_tri.normalized();
            }
        }

        self.normals = Some(normals);
    }

    /// Displace the vertices of this mesh according the given `displace_fn`.
    ///
    /// Edges longer than `max_dist` according to `distance_fn`
    /// will be recursively split until shorter before the displacement is performed.
    pub fn displace(
        &mut self,
        distance_fn: &impl Fn(Point3f, Point3f) -> Float,
        max_dist: Float,
        displace_fn: impl Fn(Point3f, Normal3f, Point2f) -> Point3f + Sync,
    ) {
        assert!(self.uv.is_some());

        // Convert faces to only triangles
        self.convert_to_only_triangles();
        // If no normals, derive them
        if self.normals.is_none() {
            self.compute_normals();
        }

        // First, refine the edges to have dists within max_dist

        // Take the tri index list, emptying it
        // (avoids borrow issues in loop below)
        let old_tri_indices = std::mem::take(&mut self.tri_indices);

        // Map of already split edges to the split point index
        let mut edge_splits = HashMap::new();
        // For every previous triangle...
        for old_tri in old_tri_indices.chunks(3) {
            self.refine_triangle_and_add(
                distance_fn,
                max_dist,
                old_tri[0],
                old_tri[1],
                old_tri[2],
                &mut edge_splits,
            );
        }

        // Then, perform the displace
        let normals = self.normals.as_ref().unwrap();
        let uv = self.uv.as_ref().unwrap();
        let new_positions: Vec<_> = parallel_map(0..self.positions.len(), |i| {
            displace_fn(self.positions[i], normals[i], uv[i])
        });

        self.positions = new_positions;

        // Derive the new normals
        self.compute_normals();
    }

    /// Refine the given triangle `(v0, v1, v2)` (recursively split the edges)
    /// so that all resulting edges are shorter than `max_dist` according to `distance_fn`,
    /// add any new vertices' positions, normals, and UVs, and add the resulting triangles
    /// to `tri_indices`.
    ///
    /// Assumes that the given triangle is not already in `tri_indices`,
    /// and will add it even if unchanged.
    ///
    /// `edge_splits` is used to record and find already added split points.
    ///
    fn refine_triangle_and_add(
        &mut self,
        distance_fn: &impl Fn(Point3f, Point3f) -> Float,
        max_dist: Float,
        v0: usize,
        v1: usize,
        v2: usize,
        edge_splits: &mut HashMap<(usize, usize), usize>,
    ) {
        // Get the vertex positions and their distances according to distance_fn
        let p0 = self.positions[v0];
        let p1 = self.positions[v1];
        let p2 = self.positions[v2];
        let d01 = distance_fn(p0, p1);
        let d12 = distance_fn(p1, p2);
        let d20 = distance_fn(p2, p0);

        // If all edges are already within max_dist, just push this triangle, done
        if d01 < max_dist && d12 < max_dist && d20 < max_dist {
            self.tri_indices.push(v0);
            self.tri_indices.push(v1);
            self.tri_indices.push(v2);
            return;
        }

        // Order the three verts so that the first two have the longest edge
        let (va, vb, vc) = if d01 >= d12 && d01 >= d20 {
            (v0, v1, v2)
        } else if d12 >= d01 && d12 >= d20 {
            (v1, v2, v0)
        } else {
            (v2, v0, v1)
        };

        // Pair of vertices forming the edge to be split.
        // For use in edge_splits map, always order by (lesser, greater) index.
        let edge = if va < vb { (va, vb) } else { (vb, va) };

        let v_mid;
        if let Some(prev_v_mid) = edge_splits.get(&edge) {
            // If this has already been split, use the existing mid vertex's index.
            v_mid = *prev_v_mid;
        } else {
            // Otherwise, push a new vert pos,
            // which is the midpoint of the edge
            let (pa, pb) = (self.positions[va], self.positions[vb]);
            v_mid = self.positions.len();
            self.positions.push((pa + pb) / 2.0);

            // Record the split in the map
            edge_splits.insert(edge, v_mid);

            // If there are normals...
            if let Some(normals) = &mut self.normals {
                // The new vert's normal is also the average
                // (summed and normalized)
                let mut n_mid = normals[va] + normals[vb];
                if n_mid.length_squared() > 0.0 {
                    n_mid = n_mid.normalized();
                }
                normals.push(n_mid);
            }

            // If there are UVs...
            if let Some(uv) = &mut self.uv {
                // The new vert's uv is also the midpoint
                uv.push((uv[va] + uv[vb]) / 2.0);
            }
        }

        // Recursively refine the new two triangles
        self.refine_triangle_and_add(distance_fn, max_dist, va, v_mid, vc, edge_splits);
        self.refine_triangle_and_add(distance_fn, max_dist, v_mid, vb, vc, edge_splits);
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
