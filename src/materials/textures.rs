use delegate::delegate;
use enum_dispatch::enum_dispatch;

use crate::{
    core::Float,
    materials::{
        mappings::{TextureMapping2DEnum, TextureMapping3DEnum},
        TextureMapping2D, TextureMapping3D,
    },
    sampling::spectrum::{SampledSpectrum, SampledWavelengths, SpectrumEnum},
};

use super::TextureEvalContext;

#[enum_dispatch]
#[derive(Debug)]
pub enum TextureEnum {
    Float(FloatTextureEnum),
    Spectrum(SpectrumTextureEnum),
}

#[enum_dispatch]
#[derive(Debug)]
pub enum FloatTextureEnum {
    Constant(ConstantFloatTexture),
    Checkerboard(CheckerboardFloatTexture),
}

impl FloatTextureEnum {
    delegate! {
        #[through(FloatTexture)]
        to self {
            pub fn eval(&self, ctx: &TextureEvalContext) -> Float;
        }
    }
}

#[enum_dispatch(FloatTextureEnum)]
pub trait FloatTexture {
    fn eval(&self, ctx: &TextureEvalContext) -> Float;
}

#[enum_dispatch]
#[derive(Debug)]
pub enum SpectrumTextureEnum {
    Constant(ConstantSpectrumTexture),
    Checkerboard(CheckerboardSpectrumTexture),
}

impl SpectrumTextureEnum {
    delegate! {
        #[through(SpectrumTexture)]
        to self {
            pub fn eval(&self, ctx: &TextureEvalContext, lambda: &SampledWavelengths) -> SampledSpectrum;
        }
    }
}

#[enum_dispatch(SpectrumTextureEnum)]
pub trait SpectrumTexture {
    fn eval(&self, ctx: &TextureEvalContext, lambda: &SampledWavelengths) -> SampledSpectrum;
}

#[derive(Debug)]
pub struct ConstantFloatTexture {
    value: Float,
}

impl FloatTexture for ConstantFloatTexture {
    fn eval(&self, _ctx: &TextureEvalContext) -> Float {
        self.value
    }
}

impl ConstantFloatTexture {
    pub fn new(value: Float) -> Self {
        Self { value }
    }
}

#[derive(Debug)]
pub struct ConstantSpectrumTexture {
    value: SpectrumEnum,
}

impl SpectrumTexture for ConstantSpectrumTexture {
    fn eval(&self, _ctx: &TextureEvalContext, lambda: &SampledWavelengths) -> SampledSpectrum {
        self.value.sample(lambda)
    }
}

impl ConstantSpectrumTexture {
    pub fn new(value: SpectrumEnum) -> Self {
        Self { value }
    }
}

#[derive(Debug)]
pub struct CheckerboardFloatTexture {
    tex: [Box<FloatTextureEnum>; 2],
    mapping_2d: Option<TextureMapping2DEnum>,
    mapping_3d: Option<TextureMapping3DEnum>,
}

impl FloatTexture for CheckerboardFloatTexture {
    fn eval(&self, ctx: &TextureEvalContext) -> Float {
        let w = if let Some(mapping_2d) = &self.mapping_2d {
            checkerboard_2d(ctx, mapping_2d)
        } else if let Some(mapping_3d) = &self.mapping_3d {
            checkerboard_3d(ctx, mapping_3d)
        } else {
            unreachable!()
        };

        let t0 = if w != 1.0 { self.tex[0].eval(ctx) } else { 0.0 };
        let t1 = if w != 0.0 { self.tex[1].eval(ctx) } else { 0.0 };

        (1.0 - w) * t0 + w * t1
    }
}

impl CheckerboardFloatTexture {
    pub fn new_2d(tex: [FloatTextureEnum; 2], mapping_2d: TextureMapping2DEnum) -> Self {
        Self {
            tex: tex.map(Box::new),
            mapping_2d: Some(mapping_2d),
            mapping_3d: None,
        }
    }

    pub fn new_3d(tex: [FloatTextureEnum; 2], mapping_3d: TextureMapping3DEnum) -> Self {
        Self {
            tex: tex.map(Box::new),
            mapping_2d: None,
            mapping_3d: Some(mapping_3d),
        }
    }
}

#[derive(Debug)]
pub struct CheckerboardSpectrumTexture {
    tex: [Box<SpectrumTextureEnum>; 2],
    mapping_2d: Option<TextureMapping2DEnum>,
    mapping_3d: Option<TextureMapping3DEnum>,
}

impl SpectrumTexture for CheckerboardSpectrumTexture {
    fn eval(&self, ctx: &TextureEvalContext, lambda: &SampledWavelengths) -> SampledSpectrum {
        let w = if let Some(mapping_2d) = &self.mapping_2d {
            checkerboard_2d(ctx, mapping_2d)
        } else if let Some(mapping_3d) = &self.mapping_3d {
            checkerboard_3d(ctx, mapping_3d)
        } else {
            panic!("neither 2D nor 3D mapping is set")
        };

        let t0 = if w != 1.0 {
            self.tex[0].eval(ctx, lambda)
        } else {
            SampledSpectrum::with_single_value(0.0)
        };
        let t1 = if w != 0.0 {
            self.tex[1].eval(ctx, lambda)
        } else {
            SampledSpectrum::with_single_value(0.0)
        };

        (1.0 - w) * t0 + w * t1
    }
}

impl CheckerboardSpectrumTexture {
    pub fn new_2d(tex: [SpectrumTextureEnum; 2], mapping_2d: TextureMapping2DEnum) -> Self {
        Self {
            tex: tex.map(Box::new),
            mapping_2d: Some(mapping_2d),
            mapping_3d: None,
        }
    }

    pub fn new_3d(tex: [SpectrumTextureEnum; 2], mapping_3d: TextureMapping3DEnum) -> Self {
        Self {
            tex: tex.map(Box::new),
            mapping_2d: None,
            mapping_3d: Some(mapping_3d),
        }
    }
}

fn checkerboard_2d(ctx: &TextureEvalContext, mapping_2d: &impl TextureMapping2D) -> Float {
    // Integrate product of 2D checkerboard function and triangle filter
    let tex_coord = mapping_2d.map(ctx);
    let ds = 1.5 * tex_coord.dsdx.abs().max(tex_coord.dsdy.abs());
    let dt = 1.5 * tex_coord.dtdx.abs().max(tex_coord.dtdy.abs());

    0.5 - checkboard_1d_filtered(tex_coord.st[0], ds) * checkboard_1d_filtered(tex_coord.st[1], dt)
        / 2.0
}

fn checkerboard_3d(ctx: &TextureEvalContext, mapping_3d: &impl TextureMapping3D) -> Float {
    let tex_coord = mapping_3d.map(ctx);
    let dx = 1.5 * tex_coord.dpdx.x().abs().max(tex_coord.dpdy.x().abs());
    let dy = 1.5 * tex_coord.dpdx.y().abs().max(tex_coord.dpdy.y().abs());
    let dz = 1.5 * tex_coord.dpdx.z().abs().max(tex_coord.dpdy.z().abs());

    0.5 - 0.5
        * checkboard_1d_filtered(tex_coord.p.x(), dx)
        * checkboard_1d_filtered(tex_coord.p.y(), dy)
        * checkboard_1d_filtered(tex_coord.p.z(), dz)
}

fn checkerboard_1d(x: Float) -> Float {
    let y = x / 2.0 - (x / 2.0).floor() - 0.5;
    x / 2.0 + y * (1.0 - 2.0 * y.abs())
}

// Box filter
fn checkboard_1d_filtered(x: Float, radius: Float) -> Float {
    if (x - radius).floor() == (x + radius).floor() {
        (1 - 2 * (x.floor() as i32 & 1)) as Float
    } else {
        (checkerboard_1d(x + radius) - 2.0 * checkerboard_1d(x) + checkerboard_1d(x - radius))
            / (radius * radius)
    }
}
