use std::sync::{Arc, OnceLock};

use crate::{
    core::{
        spherical_triangle_area, Bounds3f, DirectionCone, Float, Normal3f, Point2f, Point3f, Ray,
        SampleInteraction, Transform, Vec3f,
    },
    memory::{
        NORMAL3F_BUFFER_CACHE, POINT2F_BUFFER_CACHE, POINT3F_BUFFER_CACHE, USIZE_BUFFER_CACHE,
    },
};

use super::{Shape, ShapeIntersection, ShapeSample, ShapeSampleContext};

#[derive(Clone, Debug)]
pub struct Triangle {
    mesh_idx: usize,
    tri_idx: usize,
}

impl Triangle {
    const MIN_SPHERICAL_SAMPLE_AREA: Float = 3e-4;
    const MAX_SPHERICAL_SAMPLE_AREA: Float = 6.22;

    pub fn new(mesh_idx: usize, tri_idx: usize) -> Self {
        Self { mesh_idx, tri_idx }
    }

    fn mesh(&self) -> &TriangleMesh {
        TriangleMesh::get(self.mesh_idx).unwrap()
    }

    pub fn mesh_positions(&self) -> (Point3f, Point3f, Point3f) {
        self.mesh().positions(self.tri_idx)
    }

    pub fn mesh_vertex_normals(&self) -> Option<(Normal3f, Normal3f, Normal3f)> {
        self.mesh().vertex_normals(self.tri_idx)
    }

    pub fn mesh_uvs(&self) -> Option<(Point2f, Point2f, Point2f)> {
        self.mesh().uvs(self.tri_idx)
    }

    pub fn solid_angle(&self, p: Point3f) -> Float {
        // Get triangle vertices
        let (p0, p1, p2) = self.mesh_positions();

        spherical_triangle_area(
            (p0 - p).normalized(),
            (p1 - p).normalized(),
            (p2 - p).normalized(),
        )
    }
}

impl Shape for Triangle {
    fn bounds(&self) -> Bounds3f {
        // Get triangle vertices
        let (p0, p1, p2) = self.mesh_positions();
        // Bounds is bounding box of the three points
        Bounds3f::new(p0, p1).union_point(p2)
    }

    fn normal_bounds(&self) -> DirectionCone {
        // Get triangle vertices
        let (p0, p1, p2) = self.mesh_positions();

        let mut n = Normal3f::from((p1 - p0).cross(p2 - p0)).normalized();

        if let Some((n0, n1, n2)) = self.mesh_vertex_normals() {
            // Ensure correct orientation of geometric normal for normal bounds
            let ns = n0 + n1 + n2;
            n = n.face_forward(ns.into());
        } else if self.mesh().reverse_orientation ^ self.mesh().transform_swaps_handedness {
            // Flip if ReverseOrientation specified XOR transform swaps handedness
            n *= -1.0;
        }

        DirectionCone::from_dir(n.into())
    }

    fn intersect(&self, ray: &Ray, t_max: Option<Float>) -> Option<ShapeIntersection> {
        todo!()
    }

    fn intersect_p(&self, ray: &Ray, t_max: Option<Float>) -> bool {
        todo!()
    }

    fn area(&self) -> Float {
        // Get triangle vertices
        let (p0, p1, p2) = self.mesh_positions();
        // Standard formula for triangle
        0.5 * (p1 - p0).cross(p2 - p0).length()
    }

    fn sample(&self, u: Point2f) -> Option<ShapeSample> {
        todo!()
    }

    fn sample_with_context(&self, ctx: &ShapeSampleContext, mut u: Point2f) -> Option<ShapeSample> {
        todo!()
    }

    fn pdf(&self, interaction: &SampleInteraction) -> Float {
        todo!()
    }

    fn pdf_with_context(&self, ctx: &ShapeSampleContext, wi: Vec3f) -> Float {
        todo!()
    }
}

#[derive(Debug)]
pub struct TriangleMesh {
    pub indices: Arc<Vec<usize>>,
    /// Vertex positions in render space.
    pub positions: Arc<Vec<Point3f>>,
    /// Per-vertex normals in render space, if any.
    pub normals: Option<Arc<Vec<Normal3f>>>,
    /// Vertex UVs, if any.
    pub uv: Option<Arc<Vec<Point2f>>>,
    pub reverse_orientation: bool,
    pub transform_swaps_handedness: bool,
}

static MESHES: OnceLock<Vec<TriangleMesh>> = OnceLock::new();

impl TriangleMesh {
    pub fn new(
        render_from_obj: &Transform,
        reverse_orientation: bool,
        indices: Vec<usize>,
        mut positions: Vec<Point3f>,
        normals: Option<Vec<Normal3f>>,
        uv: Option<Vec<Point2f>>,
    ) -> Self {
        assert_eq!(
            indices.len() % 3,
            0,
            "Number of vertex indices for a triangle mesh must be multiple of 3, but got {}",
            indices.len()
        );

        // Lookup indices in cache
        let indices = USIZE_BUFFER_CACHE.lookup_or_add(indices);

        let positions = {
            // Transform positions to render space
            for p in positions.iter_mut() {
                *p = render_from_obj * *p;
            }
            POINT3F_BUFFER_CACHE.lookup_or_add(positions)
        };

        let normals = normals.map(|mut vec| {
            // Num must match num of indices
            assert_eq!(vec.len(), indices.len());
            // Transform normals to render space
            for n in vec.iter_mut() {
                *n = render_from_obj * *n;
                if reverse_orientation {
                    *n = -*n;
                }
            }
            NORMAL3F_BUFFER_CACHE.lookup_or_add(vec)
        });

        let uv = uv.map(|vec| {
            // Num must match num of indices
            assert_eq!(vec.len(), indices.len());
            POINT2F_BUFFER_CACHE.lookup_or_add(vec)
        });

        Self {
            indices,
            positions,
            normals,
            uv,
            reverse_orientation,
            transform_swaps_handedness: render_from_obj.swaps_handedness(),
        }
    }

    pub fn get(idx: usize) -> Option<&'static Self> {
        MESHES
            .get()
            .expect("Should not try to get() a mesh from storage before storage's been initialized")
            .get(idx)
    }

    pub fn init_mesh_data(all_meshes: Vec<TriangleMesh>) {
        MESHES
            .set(all_meshes)
            .expect("Mesh storage shouldn't be set more than once")
    }

    pub fn positions(&self, tri_idx: usize) -> (Point3f, Point3f, Point3f) {
        let verts = &self.indices[3 * tri_idx..3 * tri_idx + 3];
        let p0 = self.positions[verts[0]];
        let p1 = self.positions[verts[1]];
        let p2 = self.positions[verts[2]];
        (p0, p1, p2)
    }

    pub fn vertex_normals(&self, tri_idx: usize) -> Option<(Normal3f, Normal3f, Normal3f)> {
        if let Some(vert_n) = &self.normals {
            let verts = &self.indices[3 * tri_idx..3 * tri_idx + 3];
            let n0 = vert_n[verts[0]];
            let n1 = vert_n[verts[1]];
            let n2 = vert_n[verts[2]];
            Some((n0, n1, n2))
        } else {
            None
        }
    }

    pub fn uvs(&self, tri_idx: usize) -> Option<(Point2f, Point2f, Point2f)> {
        if let Some(uv) = &self.uv {
            let verts = &self.indices[3 * tri_idx..3 * tri_idx + 3];
            let uv0 = uv[verts[0]];
            let uv1 = uv[verts[1]];
            let uv2 = uv[verts[2]];
            Some((uv0, uv1, uv2))
        } else {
            None
        }
    }

    pub fn num_triangles(&self) -> usize {
        self.indices.len() / 3
    }
}
