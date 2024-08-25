use std::sync::OnceLock;

use itertools::iproduct;

use crate::{
    geometry::{Bounds3f, DirectionCone, Ray, SampleInteraction, SurfaceInteraction, Transform},
    math::{
        difference_of_products, gamma, lerp, solve_quadratic, Normal3f, Point2f, Point3f, Point3fi,
        SquareMatrix, Tuple, Vec3f,
    },
    sampling::routines::{bilinear_pdf, sample_bilinear, PiecewiseConstant2D},
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
        let (p00, p10, p01, p11) = mesh.positions(blp_idx);

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

    pub fn mesh_positions(&self) -> (Point3f, Point3f, Point3f, Point3f) {
        self.mesh().positions(self.blp_idx)
    }

    pub fn mesh_vertex_normals(&self) -> Option<(Normal3f, Normal3f, Normal3f, Normal3f)> {
        self.mesh().vertex_normals(self.blp_idx)
    }

    pub fn mesh_uvs(&self) -> Option<(Point2f, Point2f, Point2f, Point2f)> {
        self.mesh().uvs(self.blp_idx)
    }

    pub fn interaction_from_intersection(
        mesh: &BilinearPatchMesh,
        blp_idx: usize,
        uv: Point2f,
        time: Float,
        outgoing: Vec3f,
    ) -> SurfaceInteraction {
        #![allow(non_snake_case)]

        // Get bilinear patch vertices
        let (p00, p10, p01, p11) = mesh.positions(blp_idx);

        // Compute patch point p, dp/du, dp/dv for (u, v)
        let p = lerp(lerp(p00, p01, uv[1]), lerp(p10, p11, uv[1]), uv[0]);
        let dpdu: Vec3f = (lerp(p10, p11, uv[1]) - lerp(p00, p01, uv[1])).into();
        let dpdv: Vec3f = (lerp(p01, p11, uv[0]) - lerp(p00, p10, uv[0])).into();

        // TODO: Skipping this part for now
        // Compute (s, t) texcoords at patch (u, v)
        /*
        let st = uv;
        let duds = 1.0;
        let dudt = 0.0;
        let dvds = 0.0;
        let dvdt = 1.0;
        // if Some mesh.uv...
        */

        // Find partial derivatives dndu and dndv for patch
        let d2p_duu = Vec3f::ZERO;
        let d2p_dvv = Vec3f::ZERO;
        let d2p_duv = (p00 - p01) + (p11 - p10);
        // Compute coeffs for fundamental forms
        let E = dpdu.dot(dpdu);
        let F = dpdu.dot(dpdv);
        let G = dpdv.dot(dpdv);
        let n = dpdu.cross(dpdu).normalized();
        let e = n.dot(d2p_duu);
        let f = n.dot(d2p_duv);
        let g = n.dot(d2p_dvv);
        // Compute dn/du and dn/dv from coeffs
        let EGF2 = difference_of_products(E, G, F, F);
        let inv_EGF2 = if EGF2 != 0.0 { 1.0 / EGF2 } else { 0.0 };
        let dndu: Normal3f =
            ((f * F - e * G) * inv_EGF2 * dpdu + (e * F - f * E) * inv_EGF2 * dpdv).into();
        let dndv: Normal3f =
            ((g * F - f * G) * inv_EGF2 * dpdu + (f * F - g * E) * inv_EGF2 * dpdv).into();
        // TODO: Update dn/du and dn/dv to account for (s, t) parameterization

        // Initialize intersection point error
        let p_abs_sum = p00.abs() + p10.abs() + p01.abs() + p11.abs();
        let p_err = gamma(6) * Vec3f::from(p_abs_sum);

        let flip_normal = mesh.reverse_orientation ^ mesh.transform_swaps_handedness;
        let mut isect = SurfaceInteraction::builder()
            .pi(Point3fi::new_fi(p, p_err))
            .wo(outgoing)
            .dpdu(dpdu)
            .dpdv(dpdv)
            .dndu(dndu)
            .dndv(dndv)
            .time(time)
            .flip_normal(flip_normal)
            .build()
            .unwrap();

        // Compute patch shading normal if necessary
        if let Some((n00, n10, n01, n11)) = mesh.vertex_normals(blp_idx) {
            let mut ns = lerp(lerp(n00, n01, uv[1]), lerp(n10, n11, uv[1]), uv[0]);
            if ns.length_squared() > 0.0 {
                ns = ns.normalized();
                // Set shading geometry for patch intersection
                let dndu = lerp(n10, n11, uv[1]) - lerp(n00, n01, uv[1]);
                let dndv = lerp(n01, n11, uv[0]) - lerp(n00, n10, uv[0]);
                // TODO: Update dn/du and dn/dv to account for (s, t) parameterization
                let rotation = Transform::rotate_from_to(isect.n.normalized().into(), ns.into());
                isect.set_shading_geometry(ns, &rotation * dpdu, rotation * dpdv, dndu, dndv, true);
            }
        }

        isect
    }
}

impl Shape for BilinearPatch {
    fn bounds(&self) -> Bounds3f {
        // Get patch vertices
        let (p00, p10, p01, p11) = self.mesh_positions();
        // Bounds is bounding box of the four corners
        Bounds3f::new(p00, p01).union(Bounds3f::new(p10, p11))
    }

    fn normal_bounds(&self) -> DirectionCone {
        // Get patch vertices
        let (p00, p10, p01, p11) = self.mesh_positions();
        // If patch is a triangle, return bounds for single surface normal
        if p00 == p10 || p10 == p11 || p01 == p11 || p00 == p11 {
            let dpdu = lerp(p10, p11, 0.5) - lerp(p00, p01, 0.5);
            let dpdv = lerp(p01, p11, 0.5) - lerp(p00, p10, 0.5);
            let mut n: Normal3f = dpdu.cross(dpdv).normalized().into();

            if let Some((vn00, vn10, vn01, vn11)) = self.mesh_vertex_normals() {
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
        if let Some((vn00, vn10, vn01, vn11)) = self.mesh_vertex_normals() {
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
        // Get positions
        let (p00, p10, p01, p11) = self.mesh_positions();

        let (uv, mut pdf) = if let Some(distrib) = &self.mesh().image_distribution {
            todo!()
        } else if !self.mesh().is_rectangle() {
            // Sample patch (u, v) with approx uniform area sampling

            // Init w array with differential area at patch corners
            let w = [
                (p10 - p00).cross(p01 - p00).length(),
                (p10 - p00).cross(p11 - p10).length(),
                (p01 - p00).cross(p11 - p01).length(),
                (p11 - p10).cross(p11 - p01).length(),
            ];

            let uv = sample_bilinear(u, &w);
            let pdf = bilinear_pdf(uv, &w);
            (uv, pdf)
        } else {
            (u, 1.0)
        };

        // Compute patch geometric quantities at sampled (u, v)

        // Compute p, dp/du, dp/dv
        let pu0 = lerp(p00, p01, uv[1]);
        let pu1 = lerp(p10, p11, uv[1]);
        let p = lerp(pu0, pu1, uv[0]);
        let dpdu = pu1 - pu0;
        let dpdv = lerp(p01, p11, uv[0]) - lerp(p00, p10, uv[0]);
        if dpdu.length_squared() == 0.0 || dpdv.length_squared() == 0.0 {
            return None;
        }

        let st = uv;
        if let Some(uv) = self.mesh_uvs() {
            todo!()
        }

        // Compute surface normal for sampled (u, v)
        let mut n: Normal3f = dpdu.cross(dpdv).normalized().into();
        // Flip normal if necessary
        if let Some((n00, n10, n01, n11)) = self.mesh_vertex_normals() {
            let ns = lerp(lerp(n00, n01, uv[1]), lerp(n10, n11, uv[1]), uv[0]);
            n = n.face_forward(ns.into());
        } else if self.mesh().reverse_orientation ^ self.mesh().transform_swaps_handedness {
            n = -n;
        }

        // Compute error for (u, v)
        let p_abs_sum = p00.abs() + p01.abs() + p10.abs() + p11.abs();
        let p_err = gamma(6) * Vec3f::from(p_abs_sum);

        // Return sample
        let intr = SampleInteraction::new(Point3fi::new_fi(p, p_err), None, n, st);
        pdf /= dpdu.cross(dpdv).length();
        Some(ShapeSample { intr, pdf })
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
    pub uv: Option<&'static [Point2f]>,
    pub reverse_orientation: bool,
    pub transform_swaps_handedness: bool,
    pub image_distribution: Option<PiecewiseConstant2D>,
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

    pub fn positions(&self, blp_idx: usize) -> (Point3f, Point3f, Point3f, Point3f) {
        let verts = &self.vertices[4 * blp_idx..4 * blp_idx + 4];
        let p00 = self.positions[verts[0]];
        let p10 = self.positions[verts[1]];
        let p01 = self.positions[verts[2]];
        let p11 = self.positions[verts[3]];
        (p00, p10, p01, p11)
    }

    pub fn vertex_normals(
        &self,
        blp_idx: usize,
    ) -> Option<(Normal3f, Normal3f, Normal3f, Normal3f)> {
        if let Some(vert_n) = self.normals {
            let verts = &self.vertices[4 * blp_idx..4 * blp_idx + 4];
            let n00 = vert_n[verts[0]];
            let n10 = vert_n[verts[1]];
            let n01 = vert_n[verts[2]];
            let n11 = vert_n[verts[3]];
            Some((n00, n10, n01, n11))
        } else {
            None
        }
    }

    pub fn uvs(&self, blp_idx: usize) -> Option<(Point2f, Point2f, Point2f, Point2f)> {
        if let Some(uv) = self.uv {
            let verts = &self.vertices[4 * blp_idx..4 * blp_idx + 4];
            let uv00 = uv[verts[0]];
            let uv10 = uv[verts[1]];
            let uv01 = uv[verts[2]];
            let uv11 = uv[verts[3]];
            Some((uv00, uv10, uv01, uv11))
        } else {
            None
        }
    }

    pub fn is_rectangle(&self) -> bool {
        todo!()
    }
}

pub struct BilinearIntersection {
    pub uv: Point2f,
    pub t: Float,
}
