use thiserror::Error;

#[derive(Debug, Error)]
pub enum BuilderError {
    #[error("invalid argument(s) passed to builder: {0}")]
    ValidationError(String),
}
