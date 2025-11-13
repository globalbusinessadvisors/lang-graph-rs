//! Checkpointer trait and types

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use langgraph_core::{Result, State};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Checkpoint metadata
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CheckpointMetadata {
    /// Unique checkpoint ID
    pub id: String,
    /// Thread/session ID for grouping related checkpoints
    pub thread_id: String,
    /// Parent checkpoint ID (if any)
    pub parent_id: Option<String>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Custom metadata
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

impl CheckpointMetadata {
    /// Create new checkpoint metadata
    pub fn new(thread_id: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            thread_id: thread_id.into(),
            parent_id: None,
            created_at: Utc::now(),
            extra: HashMap::new(),
        }
    }

    /// Create with parent checkpoint
    pub fn with_parent(thread_id: impl Into<String>, parent_id: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            thread_id: thread_id.into(),
            parent_id: Some(parent_id.into()),
            created_at: Utc::now(),
            extra: HashMap::new(),
        }
    }

    /// Add custom metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.extra.insert(key.into(), value);
        self
    }
}

/// A checkpoint containing state and metadata
#[derive(Debug, Clone, Serialize)]
#[serde(bound = "S: State")]
pub struct Checkpoint<S: State> {
    /// Checkpoint metadata
    pub metadata: CheckpointMetadata,
    /// Saved state
    pub state: S,
}

// Manual Deserialize implementation to avoid type inference issues
impl<'de, S> Deserialize<'de> for Checkpoint<S>
where
    S: State,
{
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct CheckpointHelper<S> {
            metadata: CheckpointMetadata,
            state: S,
        }

        let helper = CheckpointHelper::<S>::deserialize(deserializer)?;
        Ok(Checkpoint {
            metadata: helper.metadata,
            state: helper.state,
        })
    }
}

impl<S: State> Checkpoint<S> {
    /// Create a new checkpoint
    pub fn new(thread_id: impl Into<String>, state: S) -> Self {
        Self {
            metadata: CheckpointMetadata::new(thread_id),
            state,
        }
    }

    /// Create checkpoint with parent
    pub fn with_parent(
        thread_id: impl Into<String>,
        parent_id: impl Into<String>,
        state: S,
    ) -> Self {
        Self {
            metadata: CheckpointMetadata::with_parent(thread_id, parent_id),
            state,
        }
    }
}

/// Checkpointer trait for persisting and retrieving state
#[async_trait]
pub trait Checkpointer<S>: Send + Sync
where
    S: State,
{
    /// Save a checkpoint
    async fn save(&self, checkpoint: Checkpoint<S>) -> Result<String>;

    /// Load a checkpoint by ID
    async fn load(&self, checkpoint_id: &str) -> Result<Option<Checkpoint<S>>>;

    /// Load the latest checkpoint for a thread
    async fn load_latest(&self, thread_id: &str) -> Result<Option<Checkpoint<S>>>;

    /// List all checkpoints for a thread
    async fn list(&self, thread_id: &str) -> Result<Vec<CheckpointMetadata>>;

    /// List all checkpoints for a thread with pagination
    async fn list_paginated(
        &self,
        thread_id: &str,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<CheckpointMetadata>>;

    /// Delete a checkpoint
    async fn delete(&self, checkpoint_id: &str) -> Result<bool>;

    /// Delete all checkpoints for a thread
    async fn delete_thread(&self, thread_id: &str) -> Result<usize>;

    /// Get checkpoint count for a thread
    async fn count(&self, thread_id: &str) -> Result<usize>;

    /// Search checkpoints by metadata
    async fn search(
        &self,
        thread_id: &str,
        metadata_filter: HashMap<String, serde_json::Value>,
    ) -> Result<Vec<CheckpointMetadata>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_metadata() {
        let meta = CheckpointMetadata::new("thread-1");
        assert_eq!(meta.thread_id, "thread-1");
        assert!(meta.parent_id.is_none());
        assert!(!meta.id.is_empty());
    }

    #[test]
    fn test_checkpoint_metadata_with_parent() {
        let meta = CheckpointMetadata::with_parent("thread-1", "parent-1");
        assert_eq!(meta.thread_id, "thread-1");
        assert_eq!(meta.parent_id, Some("parent-1".to_string()));
    }
}
