use std::sync::Arc;

use super::medium::MediumEnum;

#[derive(Clone, Debug)]
pub struct MediumInterface {
    pub inside: Arc<MediumEnum>,
    pub outside: Arc<MediumEnum>,
}

impl MediumInterface {
    pub fn is_transition(&self) -> bool {
        self.inside != self.outside
    }
}
