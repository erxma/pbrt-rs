use std::sync::OnceLock;

use itertools::iproduct;

use crate::{
    geometry::{Bounds3f, DirectionCone, Ray, SampleInteraction},
    math::{lerp, Normal3f, Point2f, Point3f, Vec3f},
    Float,
};

use super::{Shape, ShapeIntersection, ShapeSample, ShapeSampleContext};

pub struct BilinearPatch {
    mesh_idx: usize,
    blp_idx: usize,
    area: Float,
}

impl BilinearPatch {
    pub fn new(mesh: &BilinearPatchMesh, mesh_idx: usize, blp_idx: usize) -> Self {
        // Determine area of bilinear patch

        // Get bilinear patch vertices
        let verts = &mesh.vertices[4 * blp_idx..4 * blp_idx + 4];
        let p00 = mesh.positions[verts[0]];
        let p10 = mesh.positions[verts[1]];
        let p01 = mesh.positions[verts[2]];
        let p11 = mesh.positions[verts[3]];

        let area;
        if mesh.is_rectangle() {
            area = p00.distance(p01) * p00.distance(p10);
        } else {
            // Compute approx area of patch using Riemann sum evaled at NA x NA points
            const NA: usize = 3;

            let mut p = [[Point3f::ZERO; NA + 1]; NA + 1];
            for (i, j) in iproduct!(0..=NA, 0..=NA) {
                let u = i as Float / NA as Float;
                let v = j as Float / NA as Float;
                p[i][j] = lerp(lerp(p00, p01, v), lerp(p10, p11, v), u);
            }

            let mut riemann_sum = 0.0;
            for (i, j) in iproduct!(0..NA, 0..NA) {
                riemann_sum += 0.5
                    * (p[i + 1][j + 1] - p[i][j])
                        .cross(p[i + 1][j] - p[i][j + 1])
                        .length();
            }
            area = riemann_sum;
        }

        Self {
            mesh_idx,
            blp_idx,
            area,
        }
    }

    fn mesh(&self) -> &BilinearPatchMesh {
        BilinearPatchMesh::get(self.mesh_idx).unwrap()
    }

    fn positions(&self) -> (Point3f, Point3f, Point3f, Point3f) {
        let mesh = self.mesh();
        let verts = &mesh.vertices[4 * self.blp_idx..4 * self.blp_idx + 4];
        let p00 = mesh.positions[verts[0]];
        let p10 = mesh.positions[verts[1]];
        let p01 = mesh.positions[verts[2]];
        let p11 = mesh.positions[verts[3]];
        (p00, p10, p01, p11)
    }

    fn vertex_normals(&self) -> Option<(Normal3f, Normal3f, Normal3f, Normal3f)> {
        let mesh = self.mesh();
        if let Some(mesh_n) = mesh.normals {
            let verts = &mesh.vertices[4 * self.blp_idx..4 * self.blp_idx + 4];
            let n00 = mesh_n[verts[0]];
            let n10 = mesh_n[verts[1]];
            let n01 = mesh_n[verts[2]];
            let n11 = mesh_n[verts[3]];
            Some((n00, n10, n01, n11))
        } else {
            None
        }
    }
}

impl Shape for BilinearPatch {
    fn bounds(&self) -> Bounds3f {
        // Get patch vertices
        let (p00, p10, p01, p11) = self.positions();
        // Bounds is bounding box of the four corners
        Bounds3f::new(p00, p01).union(Bounds3f::new(p10, p11))
    }

    fn normal_bounds(&self) -> DirectionCone {
        // Get patch vertices
        let (p00, p10, p01, p11) = self.positions();
        // If patch is a triangle, return bounds for single surface normal
        if p00 == p10 || p10 == p11 || p01 == p11 || p00 == p11 {
            let dpdu = lerp(p10, p11, 0.5) - lerp(p00, p01, 0.5);
            let dpdv = lerp(p01, p11, 0.5) - lerp(p00, p10, 0.5);
            let mut n: Normal3f = dpdu.cross(dpdv).normalized().into();

            if let Some((vn00, vn10, vn01, vn11)) = self.vertex_normals() {
                let ns = (vn00 + vn01 + vn10 + vn11) / 4.0;
                n = n.face_forward(ns.into());
            } else if self.mesh().reverse_orientation ^ self.mesh().transform_swaps_handedness {
                n = -n;
            }
            return DirectionCone::from_dir(n.into());
        }

        // Compute patch normal at (0, 0)
        let mut n00: Normal3f = (p10 - p00).cross(p01 - p00).normalized().into();
        let mut n10: Normal3f = (p11 - p10).cross(p00 - p10).normalized().into();
        let mut n01: Normal3f = (p00 - p01).cross(p11 - p01).normalized().into();
        let mut n11: Normal3f = (p01 - p11).cross(p10 - p11).normalized().into();
        if let Some((vn00, vn10, vn01, vn11)) = self.vertex_normals() {
            n00 = n00.face_forward(vn00.into());
            n10 = n10.face_forward(vn10.into());
            n01 = n01.face_forward(vn01.into());
            n11 = n11.face_forward(vn11.into());
        } else if self.mesh().reverse_orientation ^ self.mesh().transform_swaps_handedness {
            n00 = -n00;
            n10 = -n10;
            n01 = -n01;
            n11 = -n11;
        }

        // Compute average normal and return bounds
        let n = (n00 + n10 + n01 + n11).normalized();
        // Cos of max angle any normal makes with average
        let cos_theta = [n.dot(n00), n.dot(n01), n.dot(n10), n.dot(n11)]
            .into_iter()
            .reduce(Float::min)
            .unwrap();

        DirectionCone::new(n.into(), cos_theta.clamp(-1.0, 1.0))
    }

    fn intersect(&self, ray: &Ray, t_max: Option<Float>) -> Option<ShapeIntersection> {
        todo!()
    }

    fn area(&self) -> Float {
        self.area
    }

    fn sample(&self, u: Point2f) -> Option<ShapeSample> {
        todo!()
    }

    fn sample_with_context(&self, ctx: &ShapeSampleContext, u: Point2f) -> Option<ShapeSample> {
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
pub struct BilinearPatchMesh {
    pub vertices: &'static [usize],
    pub positions: &'static [Point3f],
    pub normals: Option<&'static [Normal3f]>,
    pub reverse_orientation: bool,
    pub transform_swaps_handedness: bool,
}

static MESH_DATA: OnceLock<Vec<BilinearPatchMesh>> = OnceLock::new();

impl BilinearPatchMesh {
    pub fn get(idx: usize) -> Option<&'static Self> {
        MESH_DATA
            .get()
            .expect("Should not try to get() a mesh from storage before storage's been initialized")
            .get(idx)
    }

    pub fn init_mesh_data(all_meshes: Vec<BilinearPatchMesh>) {
        MESH_DATA
            .set(all_meshes)
            .expect("Mesh storage shouldn't be ")
    }

    pub fn is_rectangle(&self) -> bool {
        todo!()
    }
}
