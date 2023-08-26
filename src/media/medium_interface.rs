use super::medium::Medium;

#[derive(Clone, Copy, Debug)]
pub struct MediumInterface<'a> {
    pub inside: &'a Medium,
    pub outside: &'a Medium,
}
