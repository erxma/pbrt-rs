use std::sync::Arc;

use super::medium::Medium;

#[derive(Clone, Debug)]
pub struct MediumInterface {
    pub inside: Arc<Medium>,
    pub outside: Arc<Medium>,
}

impl MediumInterface {
    pub fn is_transition(&self) -> bool {
        self.inside != self.outside
    }
}
