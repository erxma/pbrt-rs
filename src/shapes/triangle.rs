use std::sync::{Arc, OnceLock};

use crate::{
    core::{
        gamma, spherical_triangle_area, Bounds3f, DirectionCone, Float, Normal3f, Point2f, Point3f,
        Point3fi, Ray, SampleInteraction, SurfaceInteraction, Transform, Tuple, Vec3f,
    },
    math::difference_of_products,
    memory::{
        NORMAL3F_BUFFER_CACHE, POINT2F_BUFFER_CACHE, POINT3F_BUFFER_CACHE, USIZE_BUFFER_CACHE,
    },
    sampling::routines::{
        bilinear_pdf, invert_spherical_triangle_sample, sample_bilinear, sample_spherical_triangle,
        sample_uniform_triangle,
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

    pub fn interaction_from_intersection(
        mesh: &TriangleMesh,
        tri_idx: usize,
        tri_isect: TriangleIntersection,
        time: Float,
        outgoing: Vec3f,
    ) -> SurfaceInteraction {
        todo!()
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
        let t_max = t_max.unwrap_or(Float::INFINITY);

        // Get positions
        let (p0, p1, p2) = self.mesh_positions();

        let tri_isect = intersect_triangle(ray, t_max, p0, p1, p2)?;
        let t_hit = tri_isect.t;

        let intr = Self::interaction_from_intersection(
            self.mesh(),
            self.tri_idx,
            tri_isect,
            ray.time,
            -ray.dir,
        );

        Some(ShapeIntersection { intr, t_hit })
    }

    fn intersect_p(&self, ray: &Ray, t_max: Option<Float>) -> bool {
        let t_max = t_max.unwrap_or(Float::INFINITY);

        let (p0, p1, p2) = self.mesh_positions();
        intersect_triangle(ray, t_max, p0, p1, p2).is_some()
    }

    fn area(&self) -> Float {
        // Get triangle vertices
        let (p0, p1, p2) = self.mesh_positions();
        // Standard formula for triangle
        0.5 * (p1 - p0).cross(p2 - p0).length()
    }

    fn sample(&self, u: Point2f) -> Option<ShapeSample> {
        // Get triangle vertices
        let (p0, p1, p2) = self.mesh_positions();

        // Sample point on triangle uniformly by area
        let b = sample_uniform_triangle(u);
        let p = b[0] * p0 + b[1] * p1 + b[2] * p2;

        // Compute surface normal for sampled point on triangle
        let mut n = Normal3f::from((p1 - p0).cross(p2 - p0).normalized());
        if let Some((n0, n1, n2)) = self.mesh_vertex_normals() {
            let ns = b[0] * n0 + b[1] * n1 + (1.0 - b[0] - b[1]) * n2;
            n = n.face_forward(ns.into());
        } else if self.mesh().reverse_orientation ^ self.mesh().transform_swaps_handedness {
            n *= -1.0;
        }

        // Get triangle uvs
        let (uv0, uv1, uv2) = self.mesh_uvs().unwrap_or((
            Point2f::ZERO,
            Point2f::new(1.0, 0.0),
            Point2f::new(1.0, 1.0),
        ));
        // Compute (u,v) for sampled point on triangle
        let uv_sample = b[0] * uv0 + b[1] * uv1 + b[2] * uv2;

        // Compute error bounds for sample point on triangle
        let p_abs_sum = (b[0] * p0).abs() + (b[1] * p1).abs() + ((1.0 - b[0] - b[1]) * p2).abs();
        let p_err = Vec3f::from(gamma(6) * p_abs_sum);

        // Return sample
        Some(ShapeSample {
            intr: SampleInteraction::new(Point3fi::new_fi(p, p_err), None, n, uv_sample),
            pdf: 1.0 / self.area(),
        })
    }

    fn sample_with_context(&self, ctx: &ShapeSampleContext, mut u: Point2f) -> Option<ShapeSample> {
        // Get triangle vertices
        let (p0, p1, p2) = self.mesh_positions();

        // Use uniform area sampling for numerically unstable cases
        let solid_angle = self.solid_angle(ctx.pi.midpoints());
        if (Self::MIN_SPHERICAL_SAMPLE_AREA..Self::MAX_SPHERICAL_SAMPLE_AREA).contains(&solid_angle)
        {
            // Regular case

            // Sample spherical triangle from reference point:
            // Apply warp product sampling for consine factor at ref point
            let mut pdf;
            if let Some(ns) = ctx.ns {
                // Compute cos(theta)-based weights at sample domain corners
                let rp = ctx.pi.midpoints();
                let wi = [
                    (p0 - rp).normalized(),
                    (p1 - rp).normalized(),
                    (p2 - rp).normalized(),
                ];
                let w = [
                    ns.absdot_v(wi[1]).max(0.01),
                    ns.absdot_v(wi[1]).max(0.01),
                    ns.absdot_v(wi[0]).max(0.01),
                    ns.absdot_v(wi[2]).max(0.01),
                ];
                u = sample_bilinear(u, &w);
                pdf = bilinear_pdf(u, &w);
            } else {
                pdf = 1.0;
            }

            let (b, tri_pdf) = sample_spherical_triangle(ctx.pi.midpoints(), [p0, p1, p2], u);
            if tri_pdf == 0.0 {
                return None;
            }
            pdf *= tri_pdf;

            let p = b[0] * p0 + b[1] * p1 + b[2] * p2;

            // Compute surface normal for sampled point on triangle
            let mut n = Normal3f::from((p1 - p0).cross(p2 - p0).normalized());
            if let Some((n0, n1, n2)) = self.mesh_vertex_normals() {
                let ns = b[0] * n0 + b[1] * n1 + (1.0 - b[0] - b[1]) * n2;
                n = n.face_forward(ns.into());
            } else if self.mesh().reverse_orientation ^ self.mesh().transform_swaps_handedness {
                n *= -1.0;
            }

            // Get triangle uvs
            let (uv0, uv1, uv2) = self.mesh_uvs().unwrap_or((
                Point2f::ZERO,
                Point2f::new(1.0, 0.0),
                Point2f::new(1.0, 1.0),
            ));
            // Compute (u,v) for sampled point on triangle
            let uv_sample = b[0] * uv0 + b[1] * uv1 + b[2] * uv2;

            // Compute error bounds for sampled point on triangle
            let p_abs_sum =
                (b[0] * p0).abs() + (b[1] * p1).abs() + ((1.0 - b[0] - b[1]) * p2).abs();
            let p_err = Vec3f::from(gamma(6) * p_abs_sum);

            // Return sample
            Some(ShapeSample {
                intr: SampleInteraction::new(
                    Point3fi::new_fi(p, p_err),
                    Some(ctx.time),
                    n,
                    uv_sample,
                ),
                pdf,
            })
        } else {
            // Numerically unstable case
            // Sample shape by area and compute incident direction
            let mut sample = self.sample(u)?;
            sample.intr.time = ctx.time;
            let mut wi = sample.intr.pi.midpoints() - ctx.pi.midpoints();
            if wi.length_squared() == 0.0 {
                return None;
            }
            wi = wi.normalized();

            // Convert area sampling PDF in sample to solid angle measure
            sample.pdf /= sample.intr.n.absdot_v(-wi)
                / ctx
                    .pi
                    .midpoints()
                    .distance_squared(sample.intr.pi.midpoints());
            if sample.pdf.is_infinite() {
                return None;
            }

            Some(sample)
        }
    }

    fn pdf(&self, _interaction: &SampleInteraction) -> Float {
        1.0 / self.area()
    }

    fn pdf_with_context(&self, ctx: &ShapeSampleContext, wi: Vec3f) -> Float {
        // Base on uniform area sampling for numerically unstable cases
        let solid_angle = self.solid_angle(ctx.pi.midpoints());

        if (Self::MIN_SPHERICAL_SAMPLE_AREA..Self::MAX_SPHERICAL_SAMPLE_AREA).contains(&solid_angle)
        {
            // Regular case

            let mut pdf = 1.0 / solid_angle;

            // Adjust PDF for warp product sampling of triangle cos(theta) factor
            if let Some(ns) = ctx.ns {
                // Get triangle vertices
                let (p0, p1, p2) = self.mesh_positions();

                let u = invert_spherical_triangle_sample(ctx.pi.midpoints(), [p0, p1, p2], wi);

                // Compute cos(theta)-based weights at sample domain corners
                let rp = ctx.pi.midpoints();
                let wi = [
                    (p0 - rp).normalized(),
                    (p1 - rp).normalized(),
                    (p2 - rp).normalized(),
                ];
                let w = [
                    ns.absdot_v(wi[1]).max(0.01),
                    ns.absdot_v(wi[1]).max(0.01),
                    ns.absdot_v(wi[0]).max(0.01),
                    ns.absdot_v(wi[2]).max(0.01),
                ];

                pdf *= bilinear_pdf(u, &w);
            }

            pdf
        } else {
            // Numerically unstable case

            // Intersect sample ray with shape geometry
            let ray = ctx.spawn_ray_with_dir(wi);
            let isect = self.intersect(&ray, None);

            if let Some(isect) = isect {
                // Compute PDF in solid angle measure from shape intersection point
                let pdf = (1.0 / self.area())
                    / isect.intr.n.absdot_v(-wi)
                    / ctx
                        .pi
                        .midpoints()
                        .distance_squared(isect.intr.pi.midpoints());
                if pdf.is_finite() {
                    pdf
                } else {
                    0.0
                }
            } else {
                0.0
            }
        }
    }
}

pub fn intersect_triangle(
    ray: &Ray,
    t_max: Float,
    p0: Point3f,
    p1: Point3f,
    p2: Point3f,
) -> Option<TriangleIntersection> {
    // Return no intersection if triangle is degenerate
    if (p2 - p0).cross(p1 - p0).length_squared() == 0.0 {
        return None;
    }

    // Transform triangle verts to ray coordinate space:

    // Translate verts based on ray origin
    let mut p0t = p0 - ray.o;
    let mut p1t = p1 - ray.o;
    let mut p2t = p2 - ray.o;

    // Permute components of triangle verts and ray direction
    let kz = ray.dir.abs().max_dimension();
    let kx = (kz + 1) % 3;
    let ky = (kx + 1) % 3;
    let dir = ray.dir.permute([kx, ky, kz]);
    p0t = p0t.permute([kx, ky, kz]);
    p1t = p1t.permute([kx, ky, kz]);
    p2t = p2t.permute([kx, ky, kz]);

    // Apply shear transformation to translated vert pos
    let sx = -dir.x() / dir.z();
    let sy = -dir.y() / dir.z();
    let sz = 1.0 / dir.z();
    *p0t.x_mut() += sx * p0t.z();
    *p0t.y_mut() += sy * p0t.z();
    *p1t.x_mut() += sx * p1t.z();
    *p1t.y_mut() += sy * p1t.z();
    *p2t.x_mut() += sx * p2t.z();
    *p2t.y_mut() += sy * p2t.z();

    // Compute edge function coefficients
    let e0 = difference_of_products(p1t.x(), p2t.y(), p1t.y(), p2t.x());
    let e1 = difference_of_products(p2t.x(), p0t.y(), p2t.y(), p0t.x());
    let e2 = difference_of_products(p0t.x(), p1t.y(), p0t.y(), p1t.x());

    // Perform triangle and determinant tests
    if (e0 < 0.0 || e1 < 0.0 || e2 < 0.0) && (e0 > 0.0 || e1 > 0.0 || e2 > 0.0) {
        return None;
    }
    let det = e0 + e1 + e2;
    if det == 0.0 {
        return None;
    }

    // Compute scaled hit distance to triangle and test against ray t range
    *p0t.z_mut() *= sz;
    *p1t.z_mut() *= sz;
    *p2t.z_mut() *= sz;
    let t_scaled = e0 * p0t.z() + e1 * p1t.z() + e2 * p2t.z();
    if det < 0.0 && (t_scaled >= 0.0 || t_scaled < t_max * det) {
        return None;
    } else if det > 0.0 && (t_scaled <= 0.0 || t_scaled > t_max * det) {
        return None;
    }

    // Compute barycentric coordinates and t value for intersection
    let inv_det = 1.0 / det;
    let b0 = e0 * inv_det;
    let b1 = e1 * inv_det;
    let b2 = e2 * inv_det;
    let t = t_scaled * inv_det;

    // TODO: Ensure that computed triangle t is conservatively greater than zero

    // Return intersection
    Some(TriangleIntersection { b0, b1, b2, t })
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

/// The barycentric coordiantes and the t value along the ray
/// where the intersection occurred.
pub struct TriangleIntersection {
    pub b0: Float,
    pub b1: Float,
    pub b2: Float,
    pub t: Float,
}
