use delegate::delegate;
use enum_dispatch::enum_dispatch;

use crate::Float;

//#[enum_dispatch]
#[derive(Clone, Debug)]
pub enum Filter {
    BoxFilter(BoxFilter),
}

impl Filter {
    delegate! {
        #[through(FilterTrait)]
        to self {
            pub fn integral(&self) -> Float;
        }
    }
}

#[enum_dispatch(Filter)]
trait FilterTrait {
    fn integral(&self) -> Float;
}

impl FilterTrait for Filter {
    fn integral(&self) -> Float {
        todo!()
    }
}

#[derive(Clone, Debug)]
pub struct BoxFilter {}

impl FilterTrait for BoxFilter {
    fn integral(&self) -> Float {
        todo!()
    }
}
