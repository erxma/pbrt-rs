use enum_dispatch::enum_dispatch;

use crate::{
    core::{safe_sqrt, Float},
    math::evaluate_polynomial,
    util::data::SRGB_TO_LINEAR_LUT,
};

#[enum_dispatch(ColorEncodingEnum)]
pub trait ColorEncoding {
    fn u8_to_linear(&self, value: u8) -> Float;
    fn linear_to_u8(&self, value: Float) -> u8;
    fn float_to_linear(&self, value: Float) -> Float;
}

#[enum_dispatch]
#[derive(Debug)]
pub enum ColorEncodingEnum {
    Linear(LinearColorEncoding),
    Srgb(SrgbColorEncoding),
}

#[derive(Debug)]
pub struct LinearColorEncoding;

impl ColorEncoding for LinearColorEncoding {
    #[inline]
    fn u8_to_linear(&self, value: u8) -> Float {
        value as Float / 255.0
    }

    #[inline]
    fn linear_to_u8(&self, value: Float) -> u8 {
        (value * 255.0 + 0.5).clamp(0.0, 255.0) as u8
    }

    #[inline]
    fn float_to_linear(&self, value: Float) -> Float {
        value
    }
}

#[derive(Debug)]
pub struct SrgbColorEncoding;

impl ColorEncoding for SrgbColorEncoding {
    #[inline]
    fn u8_to_linear(&self, value: u8) -> Float {
        SRGB_TO_LINEAR_LUT[value as usize]
    }

    #[inline]
    fn linear_to_u8(&self, value: Float) -> u8 {
        match value {
            ..=0.0 => 0,
            1.0.. => 255,
            _ => (255.0 * linear_to_srgb(value)).round().clamp(0.0, 255.0) as u8,
        }
    }

    #[inline]
    fn float_to_linear(&self, value: Float) -> Float {
        srgb_to_linear(value)
    }
}

#[inline]
#[allow(clippy::excessive_precision)]
fn linear_to_srgb(value: Float) -> Float {
    if value <= 0.0031308 {
        12.92 * value
    } else {
        // Minimax polynomial approximation from enoki's color.h.
        let sqrt = safe_sqrt(value);
        let p = evaluate_polynomial(
            sqrt,
            &[
                -0.0016829072605308378,
                0.03453868659826638,
                0.7642611304733891,
                2.0041169284241644,
                0.7551545191665577,
                -0.016202083165206348,
            ],
        );
        let q = evaluate_polynomial(
            sqrt,
            &[
                4.178892964897981e-7,
                -0.00004375359692957097,
                0.03467195408529984,
                0.6085338522168684,
                1.8970238036421054,
                1.0,
            ],
        );
        p / q * value
    }
}

#[inline]
#[allow(clippy::excessive_precision)]
fn srgb_to_linear(value: Float) -> Float {
    if value <= 0.04045 {
        value * (1.0 / 12.92)
    } else {
        // Minimax polynomial approximation from enoki's color.h.
        let p = evaluate_polynomial(
            value,
            &[
                -0.0163933279112946,
                -0.7386328024653209,
                -11.199318357635072,
                -47.46726633009393,
                -36.04572663838034,
            ],
        );
        let q = evaluate_polynomial(
            value,
            &[
                -0.004261480793199332,
                -19.140923959601675,
                -59.096406619244426,
                -18.225745396846637,
                1.0,
            ],
        );
        p / q * value
    }
}
