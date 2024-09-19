use winnow::{
    ascii::{float, multispace0, multispace1},
    combinator::{alt, preceded, separated, seq, trace},
    PResult, Parser,
};

use crate::core::{Float, Point3f, Vec3f};

pub enum TransformDirective {
    Identity,
    Translate(Vec3f),
    Scale(Vec3f),
    Rotate(Rotate),
    LookAt(LookAt),
}

pub struct Rotate {
    angle_deg: Float,
    axis: Vec3f,
}

pub struct LookAt {
    eye_pos: Point3f,
    look_pos: Point3f,
    up_dir: Vec3f,
}

pub fn transform_directive(input: &mut &str) -> PResult<TransformDirective> {
    let transform = alt((
        // Identity
        "Identity".map(|_| TransformDirective::Identity),
        // Translate
        preceded(("Translate", multispace1), separated(3, float::<_, Float, _>, multispace1))
            .map(|vals: Vec<_>| TransformDirective::Translate(Vec3f::new(vals[0], vals[1], vals[2]))),
        // Scale
        preceded(("Scale", multispace1), separated(3, float::<_, Float, _>, multispace1))
            .map(|vals: Vec<_>| TransformDirective::Scale(Vec3f::new(vals[0], vals[1], vals[2]))),
        // Rotate
        seq! {Rotate {
            _: ("Rotate", multispace1),
            angle_deg: float::<_,Float,_>,
            _: multispace1,
            axis: separated(3, float::<_, Float, _>, multispace1).map(|vals: Vec<_>| Vec3f::new(vals[0], vals[1], vals[2])),
        }}.map(TransformDirective::Rotate),
        // LookAt
        seq! {LookAt {
            _: ("LookAt", multispace1),
            eye_pos: separated(3, float::<_, Float, _>, multispace1).map(|vals: Vec<_>| Point3f::new(vals[0], vals[1], vals[2])),
            _: multispace1,
            look_pos: separated(3, float::<_, Float, _>, multispace1).map(|vals: Vec<_>| Point3f::new(vals[0], vals[1], vals[2])),
            _: multispace1,
            up_dir: separated(3, float::<_, Float, _>, multispace1).map(|vals: Vec<_>| Vec3f::new(vals[0], vals[1], vals[2])),
        }}.map(TransformDirective::LookAt)
    ));

    trace("transform", transform).parse_next(input)
}
