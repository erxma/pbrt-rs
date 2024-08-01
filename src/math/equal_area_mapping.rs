use crate::{Float, PI};

use super::{evaluate_polynomial, safe_sqrt, Point2f, Vec3f};

pub fn equal_area_square_to_sphere(p: Point2f) -> Vec3f {
    assert!(
        p.x() >= 0.0 && p.x() <= 1.0 && p.y() >= 0.0 && p.y() <= 1.0,
        "Given square coordinates should be within [-0.0, 1.0]^2"
    );

    // Transform p to [-1, 1]^2 and compute absolute values
    let u = 2.0 * p.x() - 1.0;
    let v = 2.0 * p.y() - 1.0;
    let up = u.abs();
    let vp = v.abs();

    // Computer radius r as signed distance from diagonal
    let signed_distance = 1.0 - up - vp;
    let d = signed_distance.abs();
    let r = 1.0 - d;

    // Computer angle phi for square to sphere mapping
    let phi = (if r == 0.0 { 1.0 } else { (vp - up) / r + 1.0 }) * PI / 4.0;

    // Find z coordinate for spherical direction
    let z = (1.0 - r * r) * signed_distance.signum();

    // Compute cos(phi) and sin(phi) for original quadrant
    let cos_phi = phi.cos() * u.signum();
    let sin_phi = phi.sin() * v.signum();

    Vec3f::new(
        cos_phi * r * safe_sqrt(2.0 - r * r),
        sin_phi * r * safe_sqrt(2.0 - r * r),
        z,
    )
}

pub fn equal_area_sphere_to_square(d: Vec3f) -> Point2f {
    assert!(d.length_squared() > 0.999 && d.length_squared() < 1.001);

    let x = d.x().abs();
    let y = d.y().abs();
    let z = d.z().abs();

    // Compute radius r
    let r = safe_sqrt(1.0 - z);

    // Compute argument to atan, detect a=0 to avoid div by 0
    let a = x.max(y);
    let mut b = x.min(y);
    b = if a == 0.0 { 0.0 } else { b / a };

    // Polynomial approximation of atan(x)*2/pi, x=b (x=[0,1])
    // The values written here exceed precision for both f32 and f64
    // and will be truncated, which is fine.
    #[allow(clippy::excessive_precision)]
    const COEFFICENTS: [Float; 7] = [
        0.406758566246788489601959989e-5,
        0.636226545274016134946890922156,
        0.61572017898280213493197203466e-2,
        -0.247333733281268944196501420480,
        0.881770664775316294736387951347e-1,
        0.419038818029165735901852432784e-1,
        -0.251390972343483509333252996350e-1,
    ];

    let mut phi = evaluate_polynomial(b, &COEFFICENTS);

    // Extend phi if input is in the range 45-90 deg (u < v)
    if x < y {
        phi = 1.0 - phi;
    }

    // Find (u, v) based on (r, phi)
    let mut v = phi * r;
    let mut u = r - v;

    // For southern hemisphere, mirror u, v
    if d.z() < 0.0 {
        (u, v) = (1.0 - v, 1.0 - u);
    }

    // Move (u, v) to correct quadrant based on signs of (x, y)
    u *= d.x().signum();
    v *= d.y().signum();

    // Transform (u, v) from [-1, 1] to [0, 1]
    Point2f::new(0.5 * (u + 1.0), 0.5 * (v + 1.0))
}

/// Handles the boundary cases of points p that are just outside of [0, 1]^2
/// (as may happen during bilinear interpolation with image texture lookups)
/// and wraps them around to the appropriate valid coordinates
/// that can be passed to [equal_area_square_to_sphere].
pub fn wrap_equal_area_square(uv: Point2f) -> Point2f {
    let mut x = uv.x();
    let mut y = uv.y();

    if x < 0.0 {
        x = -x;
        y = 1.0 - y;
    } else if x > 1.0 {
        x = 2.0 - x;
        y = 1.0 - y;
    }

    if y < 0.0 {
        x = 1.0 - x;
        y = -y;
    } else if y > 1.0 {
        x = 1.0 - x;
        y = 2.0 - y;
    }

    Point2f::new(x, y)
}
