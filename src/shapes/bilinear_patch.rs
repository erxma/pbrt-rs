use std::sync::OnceLock;

use itertools::iproduct;

use crate::{
    geometry::{Bounds3f, DirectionCone, Ray, SampleInteraction},
    math::{gamma, lerp, solve_quadratic, Normal3f, Point2f, Point3f, SquareMatrix, Tuple, Vec3f},
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

pub fn intersect_bilinear_patch(
    ray: &Ray,
    t_max: Float,
    p00: Point3f,
    p10: Point3f,
    p01: Point3f,
    p11: Point3f,
) -> Option<BilinearIntersection> {
    // Find quadratic coeffs for distance from ray to u iso-lines
    let a = (p10 - p00).cross(p01 - p11).dot(ray.dir);
    let c = (p00 - ray.o).cross(ray.dir).dot(p01 - p00);
    let b = (p10 - ray.o).cross(ray.dir).dot(p11 - p10) - (a + c);

    // Solve quadratic for patch u intersection
    let (u1, u2) = solve_quadratic(a, b, c)?;

    // Find epsilon to ensure that candidate is greater than zero
    let epsilon = gamma(10)
        * (ray.o.abs().max_component()
            + ray.dir.abs().max_component()
            + p00.abs().max_component()
            + p10.abs().max_component()
            + p01.abs().max_component()
            + p11.abs().max_component());

    // Compute v and t for first u intersection
    // Only a valid intersection if between 0 and 1
    let mut t = t_max;
    let mut u = None;
    let mut v = None;
    if (0.0..=1.0).contains(&u1) {
        // Precompute some common terms
        let u_o = lerp(p00, p10, u1);
        let u_dir = lerp(p01, p11, u1) - u_o;
        let delta_o = u_o - ray.o;
        let perp = ray.dir.cross(u_dir);
        let p2 = perp.length_squared();

        // Compute matrix determinants for v and t numerators
        let v1 = SquareMatrix::new([
            [delta_o.x(), ray.dir.x(), perp.x()],
            [delta_o.y(), ray.dir.y(), perp.y()],
            [delta_o.z(), ray.dir.z(), perp.z()],
        ])
        .determinant();
        let t1 = SquareMatrix::new([
            [delta_o.x(), u_dir.x(), perp.x()],
            [delta_o.y(), u_dir.y(), perp.y()],
            [delta_o.z(), u_dir.z(), perp.z()],
        ])
        .determinant();

        // Set u, v, t if intersection is valid
        // (may not be despite positive t due to float error)
        if t1 > p2 * epsilon && (0.0..=p2).contains(&v1) {
            u = Some(u1);
            v = Some(v1 / p2);
            t = t1 / p2;
        }
    }

    // Compute v and t for second u intersection
    // Only a valid intersection if between 0 and 1,
    // No need to repeat if it's the same intersection
    if (0.0..=1.0).contains(&u2) && u2 != u1 {
        // Precompute some common terms
        let u_o = lerp(p00, p10, u2);
        let u_dir = lerp(p01, p11, u2) - u_o;
        let delta_o = u_o - ray.o;
        let perp = ray.dir.cross(u_dir);
        let p2 = perp.length_squared();

        // Compute matrix determinants for v and t numerators
        let v2 = SquareMatrix::new([
            [delta_o.x(), ray.dir.x(), perp.x()],
            [delta_o.y(), ray.dir.y(), perp.y()],
            [delta_o.z(), ray.dir.z(), perp.z()],
        ])
        .determinant();
        let t2 = SquareMatrix::new([
            [delta_o.x(), u_dir.x(), perp.x()],
            [delta_o.y(), u_dir.y(), perp.y()],
            [delta_o.z(), u_dir.z(), perp.z()],
        ])
        .determinant()
            / p2;

        // Set u, v, t if intersection is valid, and is closer than t1
        // (may not be despite positive t due to float error)
        if t2 > p2 * epsilon && (0.0..=p2).contains(&v2) && t2 < t {
            u = Some(u2);
            v = Some(v2 / p2);
            t = t2;
        }
    }

    // Check intersection against t_max and possibly return intersection
    if t < t_max {
        // u and v should have been set
        Some(BilinearIntersection {
            uv: Point2f::new(u.unwrap(), v.unwrap()),
            t,
        })
    } else {
        None
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

pub struct BilinearIntersection {
    pub uv: Point2f,
    pub t: Float,
}
