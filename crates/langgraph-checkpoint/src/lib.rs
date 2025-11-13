//! # LangGraph Checkpoint
//!
//! Checkpointing and persistence layer for LangGraph
//!
//! Provides multiple backend implementations for state persistence:
//! - Memory: In-memory storage using DashMap
//! - SQLite: Persistent storage with HNSW vector indexing (optional)

pub mod checkpointer;
pub mod memory;

#[cfg(feature = "sqlite")]
pub mod sqlite;

pub use checkpointer::{Checkpoint, CheckpointMetadata, Checkpointer};
pub use langgraph_core::ExecutionStatus;
pub use memory::MemoryCheckpointer;

#[cfg(feature = "sqlite")]
pub use sqlite::SqliteCheckpointer;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::{
        Checkpoint, CheckpointMetadata, Checkpointer, ExecutionStatus,
        MemoryCheckpointer,
    };

    #[cfg(feature = "sqlite")]
    pub use crate::SqliteCheckpointer;
}
