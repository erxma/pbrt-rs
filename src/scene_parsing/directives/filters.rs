use crate::{
    core::Float,
    scene_parsing::common::{
        impl_from_entity, EntityDirective, FromEntity, ParseContext, PbrtParseError,
    },
};

#[derive(Clone, Debug, PartialEq)]
pub enum Filter {
    Box(BoxFilter),
    Gaussian(GaussianFilter),
    Triangle(TriangleFilter),
}

impl Default for Filter {
    fn default() -> Self {
        Self::Box(BoxFilter::default())
    }
}

impl FromEntity for Filter {
    fn from_entity(entity: EntityDirective, ctx: &ParseContext) -> Result<Self, PbrtParseError> {
        assert_eq!(entity.identifier, "Filter");

        match entity.subtype {
            "box" => BoxFilter::from_entity(entity, ctx).map(Filter::Box),
            "gaussian" => GaussianFilter::from_entity(entity, ctx).map(Filter::Gaussian),
            "triangle" => TriangleFilter::from_entity(entity, ctx).map(Filter::Triangle),
            invalid_type => Err(PbrtParseError::UnrecognizedVariant {
                entity: "Filter".to_string(),
                variant_name: invalid_type.to_owned(),
            }),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BoxFilter {
    pub x_radius: Float,
    pub y_radius: Float,
}

impl Default for BoxFilter {
    fn default() -> Self {
        Self {
            x_radius: 0.5,
            y_radius: 0.5,
        }
    }
}

impl_from_entity! {
    BoxFilter,
    has_defaults {
        "xradius" => x_radius,
        "yradius" => y_radius
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GaussianFilter {
    pub x_radius: Float,
    pub y_radius: Float,
    pub std: Float,
}

impl Default for GaussianFilter {
    fn default() -> Self {
        Self {
            x_radius: 1.5,
            y_radius: 1.5,
            std: 0.5,
        }
    }
}

impl_from_entity! {
    GaussianFilter,
    has_defaults {
        "xradius" => x_radius,
        "yradius" => y_radius,
        "sigma" => std
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TriangleFilter {
    pub x_radius: Float,
    pub y_radius: Float,
}

impl Default for TriangleFilter {
    fn default() -> Self {
        Self {
            x_radius: 2.0,
            y_radius: 2.0,
        }
    }
}

impl_from_entity! {
    TriangleFilter,
    has_defaults {
        "xradius" => x_radius,
        "yradius" => y_radius
    }
}
