//! In-memory checkpointer implementation

use crate::{Checkpoint, CheckpointMetadata, Checkpointer};
use async_trait::async_trait;
use dashmap::DashMap;
use langgraph_core::{Result, State};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

/// In-memory checkpointer using DashMap for thread-safe storage
pub struct MemoryCheckpointer<S: State> {
    /// Checkpoint storage: checkpoint_id -> Checkpoint
    checkpoints: Arc<DashMap<String, Checkpoint<S>>>,
    /// Thread index: thread_id -> Vec<checkpoint_id>
    thread_index: Arc<DashMap<String, Arc<RwLock<Vec<String>>>>>,
}

impl<S: State> MemoryCheckpointer<S> {
    /// Create a new memory checkpointer
    pub fn new() -> Self {
        Self {
            checkpoints: Arc::new(DashMap::new()),
            thread_index: Arc::new(DashMap::new()),
        }
    }

    /// Get the number of stored checkpoints
    pub fn size(&self) -> usize {
        self.checkpoints.len()
    }

    /// Clear all checkpoints
    pub fn clear(&self) {
        self.checkpoints.clear();
        self.thread_index.clear();
    }
}

impl<S: State> Default for MemoryCheckpointer<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: State> Clone for MemoryCheckpointer<S> {
    fn clone(&self) -> Self {
        Self {
            checkpoints: self.checkpoints.clone(),
            thread_index: self.thread_index.clone(),
        }
    }
}

#[async_trait]
impl<S: State> Checkpointer<S> for MemoryCheckpointer<S> {
    async fn save(&self, checkpoint: Checkpoint<S>) -> Result<String> {
        let checkpoint_id = checkpoint.metadata.id.clone();
        let thread_id = checkpoint.metadata.thread_id.clone();

        // Store checkpoint
        self.checkpoints.insert(checkpoint_id.clone(), checkpoint);

        // Update thread index
        self.thread_index
            .entry(thread_id)
            .or_insert_with(|| Arc::new(RwLock::new(Vec::new())))
            .write()
            .push(checkpoint_id.clone());

        Ok(checkpoint_id)
    }

    async fn load(&self, checkpoint_id: &str) -> Result<Option<Checkpoint<S>>> {
        Ok(self
            .checkpoints
            .get(checkpoint_id)
            .map(|entry| entry.value().clone()))
    }

    async fn load_latest(&self, thread_id: &str) -> Result<Option<Checkpoint<S>>> {
        let last_id = self.thread_index.get(thread_id).and_then(|ids| {
            let ids = ids.read();
            ids.last().cloned()
        });

        if let Some(id) = last_id {
            self.load(&id).await
        } else {
            Ok(None)
        }
    }

    async fn list(&self, thread_id: &str) -> Result<Vec<CheckpointMetadata>> {
        let checkpoint_ids = self.thread_index.get(thread_id);
        if let Some(ids) = checkpoint_ids {
            let ids = ids.read();
            let mut metadata = Vec::new();
            for id in ids.iter() {
                if let Some(checkpoint) = self.checkpoints.get(id) {
                    metadata.push(checkpoint.metadata.clone());
                }
            }
            Ok(metadata)
        } else {
            Ok(Vec::new())
        }
    }

    async fn list_paginated(
        &self,
        thread_id: &str,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<CheckpointMetadata>> {
        let checkpoint_ids = self.thread_index.get(thread_id);
        if let Some(ids) = checkpoint_ids {
            let ids = ids.read();
            let mut metadata = Vec::new();
            for id in ids.iter().skip(offset).take(limit) {
                if let Some(checkpoint) = self.checkpoints.get(id) {
                    metadata.push(checkpoint.metadata.clone());
                }
            }
            Ok(metadata)
        } else {
            Ok(Vec::new())
        }
    }

    async fn delete(&self, checkpoint_id: &str) -> Result<bool> {
        if let Some((_, checkpoint)) = self.checkpoints.remove(checkpoint_id) {
            // Remove from thread index
            let thread_id = &checkpoint.metadata.thread_id;
            if let Some(ids) = self.thread_index.get(thread_id) {
                let mut ids = ids.write();
                ids.retain(|id| id != checkpoint_id);
            }
            Ok(true)
        } else {
            Ok(false)
        }
    }

    async fn delete_thread(&self, thread_id: &str) -> Result<usize> {
        let checkpoint_ids = self.thread_index.get(thread_id);
        if let Some(ids) = checkpoint_ids {
            let ids = ids.read().clone();
            let count = ids.len();
            for id in ids {
                self.checkpoints.remove(&id);
            }
            self.thread_index.remove(thread_id);
            Ok(count)
        } else {
            Ok(0)
        }
    }

    async fn count(&self, thread_id: &str) -> Result<usize> {
        Ok(self
            .thread_index
            .get(thread_id)
            .map(|ids| ids.read().len())
            .unwrap_or(0))
    }

    async fn search(
        &self,
        thread_id: &str,
        metadata_filter: HashMap<String, serde_json::Value>,
    ) -> Result<Vec<CheckpointMetadata>> {
        let checkpoint_ids = self.thread_index.get(thread_id);
        if let Some(ids) = checkpoint_ids {
            let ids = ids.read();
            let mut metadata = Vec::new();
            for id in ids.iter() {
                if let Some(checkpoint) = self.checkpoints.get(id) {
                    let meta = &checkpoint.metadata;
                    let matches = metadata_filter.iter().all(|(key, value)| {
                        // Check both direct fields and extra metadata
                        match key.as_str() {
                            "id" => serde_json::json!(&meta.id) == *value,
                            "thread_id" => serde_json::json!(&meta.thread_id) == *value,
                            "parent_id" => {
                                meta.parent_id
                                    .as_ref()
                                    .map(|p| serde_json::json!(p) == *value)
                                    .unwrap_or(false)
                            }
                            _ => meta.extra.get(key).map(|v| v == value).unwrap_or(false),
                        }
                    });
                    if matches {
                        metadata.push(meta.clone());
                    }
                }
            }
            Ok(metadata)
        } else {
            Ok(Vec::new())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct TestState {
        value: i32,
    }

    #[tokio::test]
    async fn test_memory_checkpointer() {
        let checkpointer = MemoryCheckpointer::new();
        let state = TestState { value: 42 };
        let checkpoint = Checkpoint::new("thread-1", state.clone());

        // Save checkpoint
        let id = checkpointer.save(checkpoint).await.unwrap();
        assert!(!id.is_empty());

        // Load checkpoint
        let loaded = checkpointer.load(&id).await.unwrap().unwrap();
        assert_eq!(loaded.state.value, 42);

        // Load latest
        let latest = checkpointer.load_latest("thread-1").await.unwrap().unwrap();
        assert_eq!(latest.state.value, 42);

        // Count
        let count = checkpointer.count("thread-1").await.unwrap();
        assert_eq!(count, 1);
    }

    #[tokio::test]
    async fn test_memory_checkpointer_list() {
        let checkpointer = MemoryCheckpointer::new();

        // Save multiple checkpoints
        for i in 0..5 {
            let state = TestState { value: i };
            let checkpoint = Checkpoint::new("thread-1", state);
            checkpointer.save(checkpoint).await.unwrap();
        }

        // List all
        let list = checkpointer.list("thread-1").await.unwrap();
        assert_eq!(list.len(), 5);

        // List paginated
        let paginated = checkpointer
            .list_paginated("thread-1", 2, 1)
            .await
            .unwrap();
        assert_eq!(paginated.len(), 2);
    }

    #[tokio::test]
    async fn test_memory_checkpointer_delete() {
        let checkpointer = MemoryCheckpointer::new();
        let state = TestState { value: 42 };
        let checkpoint = Checkpoint::new("thread-1", state);

        let id = checkpointer.save(checkpoint).await.unwrap();

        // Delete checkpoint
        let deleted = checkpointer.delete(&id).await.unwrap();
        assert!(deleted);

        // Verify deleted
        let loaded = checkpointer.load(&id).await.unwrap();
        assert!(loaded.is_none());
    }

    #[tokio::test]
    async fn test_memory_checkpointer_search() {
        let checkpointer = MemoryCheckpointer::new();
        let state = TestState { value: 42 };
        let checkpoint = Checkpoint::with_parent("thread-1", "parent-1", state);

        checkpointer.save(checkpoint).await.unwrap();

        // Search by metadata
        let mut filter = HashMap::new();
        filter.insert("parent_id".to_string(), serde_json::json!("parent-1"));

        let results = checkpointer.search("thread-1", filter).await.unwrap();
        assert_eq!(results.len(), 1);
    }
}
