use itertools::iproduct;

use crate::{
    math::{lerp, Point3f},
    Float,
};

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
}

pub struct BilinearPatchMesh<'a> {
    pub vertices: &'a [usize],
    pub positions: &'a [Point3f],
}

impl BilinearPatchMesh<'_> {
    pub fn is_rectangle(&self) -> bool {
        todo!()
    }
}
