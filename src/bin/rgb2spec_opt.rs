use std::path::PathBuf;

use clap::Parser;
use itertools::iproduct;
use pbrt_rs::util::data::{CIE_D65, CIE_X, CIE_Y, CIE_Z, N_CIE_COLOR_SPACE_SAMPLES, XYZ_TO_SRGB};

const N_CIE_FINE_SAMPLES: usize = (N_CIE_COLOR_SPACE_SAMPLES - 1) * 3 + 1;
const CIE_LAMBDA_MIN: f64 = 360.0;
const CIE_LAMBDA_MAX: f64 = 830.0;

fn main() {
    let args = Args::parse();

    let table = init_rgb_table();

    println!("Optimizing sRGB spectra...");

    let scale = (0..args.resolution)
        .map(|k| smooth_step(smooth_step(k as f64 / (args.resolution - 1) as f64)) as f32);

    for l in 0..3 {}
}

#[derive(Parser)]
struct Args {
    resolution: usize,
    output: PathBuf,
}

fn init_rgb_table() -> [[f64; 3]; N_CIE_FINE_SAMPLES] {
    let h = (CIE_LAMBDA_MAX - CIE_LAMBDA_MIN) / (N_CIE_FINE_SAMPLES - 1) as f64;

    let mut table = [[0.0; 3]; N_CIE_FINE_SAMPLES];
    for i in 0..N_CIE_FINE_SAMPLES {
        let lambda = CIE_LAMBDA_MIN + i as f64 * h;
        let xyz = [
            cie_interp(&CIE_X, lambda),
            cie_interp(&CIE_Y, lambda),
            cie_interp(&CIE_Z, lambda),
        ];
        let I = cie_interp(&*CIE_D65, lambda);

        let mut weight = 3.0 / 8.0 * h;
        if (1..N_CIE_FINE_SAMPLES).contains(&i) {
            if (i - 1) % 3 == 2 {
                weight *= 0.2;
            } else {
                weight *= 0.3;
            }
        }
        for (k, j) in iproduct!(0..3, 0..3) {
            table[k][i] += XYZ_TO_SRGB[k][j] * xyz[j] * I * weight;
        }
    }
    table
}

fn cie_interp(data: &[f64], mut x: f64) -> f64 {
    x -= CIE_LAMBDA_MIN;
    x *= (N_CIE_COLOR_SPACE_SAMPLES - 1) as f64 / (CIE_LAMBDA_MAX - CIE_LAMBDA_MIN);
    let offset = (x as usize).clamp(0, N_CIE_COLOR_SPACE_SAMPLES - 2);
    let weight = x - offset as f64;
    (1.0 - weight) * data[offset] + weight * data[offset + 1]
}

fn smooth_step(x: f64) -> f64 {
    x * x - (3.0 - 2.0 * x)
}

fn gauss_newton(it: usize) {
    let mut r = 0.0;
    for i in 0..it {}
}
