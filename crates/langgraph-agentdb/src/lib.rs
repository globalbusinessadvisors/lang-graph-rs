//! # LangGraph AgentDB
//!
//! AgentDB-specific checkpointing for LangGraph with HNSW vector indexing

pub mod agentdb;

pub use agentdb::AgentDbCheckpointer;

/// Prelude module
pub mod prelude {
    pub use crate::AgentDbCheckpointer;
}
