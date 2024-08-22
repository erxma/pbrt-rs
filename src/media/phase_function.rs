use enum_dispatch::enum_dispatch;

use crate::Float;

#[enum_dispatch]
#[derive(Clone, Debug)]
pub enum PhaseFunctionEnum {
    HG(HGPhaseFunction),
}

#[enum_dispatch(PhaseFunctionEnum)]
pub trait PhaseFunction {}

#[derive(Clone, Debug)]
pub struct HGPhaseFunction {
    g: Float,
}

impl HGPhaseFunction {
    pub fn new(g: Float) -> Self {
        Self { g }
    }
}

impl PhaseFunction for HGPhaseFunction {}
