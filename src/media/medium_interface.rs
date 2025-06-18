use std::sync::Arc;

use super::medium::MediumEnum;

#[derive(Clone, Debug)]
pub struct MediumInterface {
    pub inside: Option<Arc<MediumEnum>>,
    pub outside: Option<Arc<MediumEnum>>,
}

impl MediumInterface {
    pub fn is_transition(&self) -> bool {
        self.inside != self.outside
    }
}
