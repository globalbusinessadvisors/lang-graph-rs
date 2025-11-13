//! Error types for LangGraph core

use thiserror::Error;

/// Result type alias for LangGraph operations
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur during graph operations
#[derive(Error, Debug, Clone)]
pub enum Error {
    #[error("Node not found: {0}")]
    NodeNotFound(String),

    #[error("Edge not found: from {from} to {to}")]
    EdgeNotFound { from: String, to: String },

    #[error("Invalid graph structure: {0}")]
    InvalidGraph(String),

    #[error("Node execution failed: {0}")]
    NodeExecutionFailed(String),

    #[error("State validation failed: {0}")]
    StateValidationFailed(String),

    #[error("Cycle detected in graph: {0}")]
    CycleDetected(String),

    #[error("Graph compilation failed: {0}")]
    CompilationFailed(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    #[error("Checkpoint error: {0}")]
    CheckpointError(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

impl Error {
    /// Create a new node not found error
    pub fn node_not_found(name: impl Into<String>) -> Self {
        Self::NodeNotFound(name.into())
    }

    /// Create a new edge not found error
    pub fn edge_not_found(from: impl Into<String>, to: impl Into<String>) -> Self {
        Self::EdgeNotFound {
            from: from.into(),
            to: to.into(),
        }
    }

    /// Create a new invalid graph error
    pub fn invalid_graph(msg: impl Into<String>) -> Self {
        Self::InvalidGraph(msg.into())
    }

    /// Create a new node execution failed error
    pub fn node_execution_failed(msg: impl Into<String>) -> Self {
        Self::NodeExecutionFailed(msg.into())
    }

    /// Create a new state validation failed error
    pub fn state_validation_failed(msg: impl Into<String>) -> Self {
        Self::StateValidationFailed(msg.into())
    }

    /// Create a new cycle detected error
    pub fn cycle_detected(msg: impl Into<String>) -> Self {
        Self::CycleDetected(msg.into())
    }

    /// Create a new compilation failed error
    pub fn compilation_failed(msg: impl Into<String>) -> Self {
        Self::CompilationFailed(msg.into())
    }

    /// Create a new serialization error
    pub fn serialization_error(msg: impl Into<String>) -> Self {
        Self::SerializationError(msg.into())
    }

    /// Create a new invalid operation error
    pub fn invalid_operation(msg: impl Into<String>) -> Self {
        Self::InvalidOperation(msg.into())
    }

    /// Create a new checkpoint error
    pub fn checkpoint_error(msg: impl Into<String>) -> Self {
        Self::CheckpointError(msg.into())
    }

    /// Create a new internal error
    pub fn internal(msg: impl Into<String>) -> Self {
        Self::Internal(msg.into())
    }
}
