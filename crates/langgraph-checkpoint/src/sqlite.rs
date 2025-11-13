//! SQLite checkpointer with vector indexing support

use crate::{Checkpoint, CheckpointMetadata, Checkpointer};
use async_trait::async_trait;
use langgraph_core::{Error, Result, State};
use parking_lot::Mutex;
use rusqlite::{params, Connection, OptionalExtension};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// SQLite-based checkpointer with persistent storage
pub struct SqliteCheckpointer {
    conn: Arc<Mutex<Connection>>,
}

impl SqliteCheckpointer {
    /// Create a new SQLite checkpointer with a file
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let conn = Connection::open(path).map_err(|e| Error::checkpoint_error(e.to_string()))?;
        let checkpointer = Self {
            conn: Arc::new(Mutex::new(conn)),
        };
        checkpointer.init_schema()?;
        Ok(checkpointer)
    }

    /// Create an in-memory SQLite checkpointer
    pub fn in_memory() -> Result<Self> {
        let conn =
            Connection::open_in_memory().map_err(|e| Error::checkpoint_error(e.to_string()))?;
        let checkpointer = Self {
            conn: Arc::new(Mutex::new(conn)),
        };
        checkpointer.init_schema()?;
        Ok(checkpointer)
    }

    /// Initialize database schema
    fn init_schema(&self) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute(
            r#"
            CREATE TABLE IF NOT EXISTS checkpoints (
                id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                parent_id TEXT,
                created_at TEXT NOT NULL,
                state_json TEXT NOT NULL,
                extra_json TEXT
            )
            "#,
            [],
        )
        .map_err(|e| Error::checkpoint_error(e.to_string()))?;

        conn.execute(
            r#"
            CREATE INDEX IF NOT EXISTS idx_thread_id ON checkpoints(thread_id)
            "#,
            [],
        )
        .map_err(|e| Error::checkpoint_error(e.to_string()))?;

        conn.execute(
            r#"
            CREATE INDEX IF NOT EXISTS idx_created_at ON checkpoints(created_at)
            "#,
            [],
        )
        .map_err(|e| Error::checkpoint_error(e.to_string()))?;

        // Create vector table for HNSW indexing (simplified version)
        conn.execute(
            r#"
            CREATE TABLE IF NOT EXISTS checkpoint_vectors (
                checkpoint_id TEXT PRIMARY KEY,
                embedding BLOB,
                FOREIGN KEY (checkpoint_id) REFERENCES checkpoints(id) ON DELETE CASCADE
            )
            "#,
            [],
        )
        .map_err(|e| Error::checkpoint_error(e.to_string()))?;

        Ok(())
    }

    /// Generate a simple hash-based embedding from state
    /// In production, this would use a real embedding model
    fn generate_embedding<S: State>(state: &S) -> Vec<f32> {
        let json = serde_json::to_string(state).unwrap_or_default();
        let mut embedding = vec![0.0f32; 384]; // 384 dimensions as per spec

        // Simple hash-based embedding (for demo purposes)
        for (i, byte) in json.bytes().enumerate() {
            embedding[i % 384] += byte as f32 / 255.0;
        }

        // Normalize
        let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            for val in &mut embedding {
                *val /= magnitude;
            }
        }

        embedding
    }

    /// Store embedding for a checkpoint
    fn store_embedding(&self, checkpoint_id: &str, embedding: &[f32]) -> Result<()> {
        let conn = self.conn.lock();
        let blob: Vec<u8> = embedding
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        conn.execute(
            "INSERT OR REPLACE INTO checkpoint_vectors (checkpoint_id, embedding) VALUES (?1, ?2)",
            params![checkpoint_id, blob],
        )
        .map_err(|e| Error::checkpoint_error(e.to_string()))?;

        Ok(())
    }
}

impl Clone for SqliteCheckpointer {
    fn clone(&self) -> Self {
        Self {
            conn: self.conn.clone(),
        }
    }
}

#[async_trait]
impl<S: State> Checkpointer<S> for SqliteCheckpointer {
    async fn save(&self, checkpoint: Checkpoint<S>) -> Result<String> {
        let checkpoint_id = checkpoint.metadata.id.clone();
        let thread_id = checkpoint.metadata.thread_id.clone();
        let parent_id = checkpoint.metadata.parent_id.clone();
        let created_at = checkpoint.metadata.created_at.to_rfc3339();
        let state_json = serde_json::to_string(&checkpoint.state)
            .map_err(|e| Error::serialization_error(e.to_string()))?;
        let extra_json = if checkpoint.metadata.extra.is_empty() {
            None
        } else {
            Some(
                serde_json::to_string(&checkpoint.metadata.extra)
                    .map_err(|e| Error::serialization_error(e.to_string()))?,
            )
        };

        // Generate and store embedding
        let embedding = Self::generate_embedding(&checkpoint.state);

        let conn = self.conn.lock();
        conn.execute(
            "INSERT INTO checkpoints (id, thread_id, parent_id, created_at, state_json, extra_json) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![checkpoint_id, thread_id, parent_id, created_at, state_json, extra_json],
        )
        .map_err(|e| Error::checkpoint_error(e.to_string()))?;

        drop(conn);

        // Store embedding
        self.store_embedding(&checkpoint_id, &embedding)?;

        Ok(checkpoint_id)
    }

    async fn load(&self, checkpoint_id: &str) -> Result<Option<Checkpoint<S>>> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare(
                "SELECT id, thread_id, parent_id, created_at, state_json, extra_json FROM checkpoints WHERE id = ?1",
            )
            .map_err(|e| Error::checkpoint_error(e.to_string()))?;

        let result = stmt
            .query_row(params![checkpoint_id], |row| {
                let id: String = row.get(0)?;
                let thread_id: String = row.get(1)?;
                let parent_id: Option<String> = row.get(2)?;
                let created_at: String = row.get(3)?;
                let state_json: String = row.get(4)?;
                let extra_json: Option<String> = row.get(5)?;

                Ok((id, thread_id, parent_id, created_at, state_json, extra_json))
            })
            .optional()
            .map_err(|e| Error::checkpoint_error(e.to_string()))?;

        if let Some((id, thread_id, parent_id, created_at, state_json, extra_json)) = result {
            let state: S = serde_json::from_str(&state_json)
                .map_err(|e| Error::serialization_error(e.to_string()))?;
            let extra: HashMap<String, serde_json::Value> = if let Some(json) = extra_json {
                serde_json::from_str(&json)
                    .map_err(|e| Error::serialization_error(e.to_string()))?
            } else {
                HashMap::new()
            };
            let created_at = chrono::DateTime::parse_from_rfc3339(&created_at)
                .map_err(|e| Error::internal(e.to_string()))?
                .with_timezone(&chrono::Utc);

            Ok(Some(Checkpoint {
                metadata: CheckpointMetadata {
                    id,
                    thread_id,
                    parent_id,
                    created_at,
                    extra,
                },
                state,
            }))
        } else {
            Ok(None)
        }
    }

    async fn load_latest(&self, thread_id: &str) -> Result<Option<Checkpoint<S>>> {
        let id = {
            let conn = self.conn.lock();
            let mut stmt = conn
                .prepare(
                    "SELECT id FROM checkpoints WHERE thread_id = ?1 ORDER BY created_at DESC LIMIT 1",
                )
                .map_err(|e| Error::checkpoint_error(e.to_string()))?;

            stmt.query_row(params![thread_id], |row| row.get::<_, String>(0))
                .optional()
                .map_err(|e| Error::checkpoint_error(e.to_string()))?
        };

        if let Some(id) = id {
            <Self as Checkpointer<S>>::load(self, &id).await
        } else {
            Ok(None)
        }
    }

    async fn list(&self, thread_id: &str) -> Result<Vec<CheckpointMetadata>> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare(
                "SELECT id, thread_id, parent_id, created_at, extra_json FROM checkpoints WHERE thread_id = ?1 ORDER BY created_at ASC",
            )
            .map_err(|e| Error::checkpoint_error(e.to_string()))?;

        let rows = stmt
            .query_map(params![thread_id], |row| {
                let id: String = row.get(0)?;
                let thread_id: String = row.get(1)?;
                let parent_id: Option<String> = row.get(2)?;
                let created_at: String = row.get(3)?;
                let extra_json: Option<String> = row.get(4)?;

                Ok((id, thread_id, parent_id, created_at, extra_json))
            })
            .map_err(|e| Error::checkpoint_error(e.to_string()))?;

        let mut metadata = Vec::new();
        for row in rows {
            let (id, thread_id, parent_id, created_at, extra_json) =
                row.map_err(|e| Error::checkpoint_error(e.to_string()))?;
            let extra: HashMap<String, serde_json::Value> = if let Some(json) = extra_json {
                serde_json::from_str(&json)
                    .map_err(|e| Error::serialization_error(e.to_string()))?
            } else {
                HashMap::new()
            };
            let created_at = chrono::DateTime::parse_from_rfc3339(&created_at)
                .map_err(|e| Error::internal(e.to_string()))?
                .with_timezone(&chrono::Utc);

            metadata.push(CheckpointMetadata {
                id,
                thread_id,
                parent_id,
                created_at,
                extra,
            });
        }

        Ok(metadata)
    }

    async fn list_paginated(
        &self,
        thread_id: &str,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<CheckpointMetadata>> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare(
                "SELECT id, thread_id, parent_id, created_at, extra_json FROM checkpoints WHERE thread_id = ?1 ORDER BY created_at ASC LIMIT ?2 OFFSET ?3",
            )
            .map_err(|e| Error::checkpoint_error(e.to_string()))?;

        let rows = stmt
            .query_map(params![thread_id, limit as i64, offset as i64], |row| {
                let id: String = row.get(0)?;
                let thread_id: String = row.get(1)?;
                let parent_id: Option<String> = row.get(2)?;
                let created_at: String = row.get(3)?;
                let extra_json: Option<String> = row.get(4)?;

                Ok((id, thread_id, parent_id, created_at, extra_json))
            })
            .map_err(|e| Error::checkpoint_error(e.to_string()))?;

        let mut metadata = Vec::new();
        for row in rows {
            let (id, thread_id, parent_id, created_at, extra_json) =
                row.map_err(|e| Error::checkpoint_error(e.to_string()))?;
            let extra: HashMap<String, serde_json::Value> = if let Some(json) = extra_json {
                serde_json::from_str(&json)
                    .map_err(|e| Error::serialization_error(e.to_string()))?
            } else {
                HashMap::new()
            };
            let created_at = chrono::DateTime::parse_from_rfc3339(&created_at)
                .map_err(|e| Error::internal(e.to_string()))?
                .with_timezone(&chrono::Utc);

            metadata.push(CheckpointMetadata {
                id,
                thread_id,
                parent_id,
                created_at,
                extra,
            });
        }

        Ok(metadata)
    }

    async fn delete(&self, checkpoint_id: &str) -> Result<bool> {
        let conn = self.conn.lock();
        let rows = conn
            .execute("DELETE FROM checkpoints WHERE id = ?1", params![checkpoint_id])
            .map_err(|e| Error::checkpoint_error(e.to_string()))?;

        Ok(rows > 0)
    }

    async fn delete_thread(&self, thread_id: &str) -> Result<usize> {
        let conn = self.conn.lock();
        let rows = conn
            .execute(
                "DELETE FROM checkpoints WHERE thread_id = ?1",
                params![thread_id],
            )
            .map_err(|e| Error::checkpoint_error(e.to_string()))?;

        Ok(rows)
    }

    async fn count(&self, thread_id: &str) -> Result<usize> {
        let conn = self.conn.lock();
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?1",
                params![thread_id],
                |row| row.get(0),
            )
            .map_err(|e| Error::checkpoint_error(e.to_string()))?;

        Ok(count as usize)
    }

    async fn search(
        &self,
        thread_id: &str,
        metadata_filter: HashMap<String, serde_json::Value>,
    ) -> Result<Vec<CheckpointMetadata>> {
        // For simplicity, load all and filter in memory
        // In production, this would use proper SQL WHERE clauses
        let all_metadata: Vec<CheckpointMetadata> =
            <Self as Checkpointer<S>>::list(self, thread_id).await?;

        let filtered: Vec<CheckpointMetadata> = all_metadata
            .into_iter()
            .filter(|meta| {
                metadata_filter.iter().all(|(key, value)| {
                    match key.as_str() {
                        "id" => serde_json::json!(&meta.id) == *value,
                        "thread_id" => serde_json::json!(&meta.thread_id) == *value,
                        "parent_id" => meta
                            .parent_id
                            .as_ref()
                            .map(|p| serde_json::json!(p) == *value)
                            .unwrap_or(false),
                        _ => meta.extra.get(key).map(|v| v == value).unwrap_or(false),
                    }
                })
            })
            .collect();

        Ok(filtered)
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
    async fn test_sqlite_checkpointer() {
        let checkpointer = SqliteCheckpointer::in_memory().unwrap();
        let state = TestState { value: 42 };
        let checkpoint = Checkpoint::new("thread-1", state.clone());

        // Save checkpoint
        let id = <SqliteCheckpointer as Checkpointer<TestState>>::save(&checkpointer, checkpoint)
            .await
            .unwrap();
        assert!(!id.is_empty());

        // Load checkpoint
        let loaded: Option<Checkpoint<TestState>> =
            <SqliteCheckpointer as Checkpointer<TestState>>::load(&checkpointer, &id)
                .await
                .unwrap();
        assert_eq!(loaded.unwrap().state.value, 42);

        // Load latest
        let latest: Option<Checkpoint<TestState>> =
            <SqliteCheckpointer as Checkpointer<TestState>>::load_latest(&checkpointer, "thread-1")
                .await
                .unwrap();
        assert_eq!(latest.unwrap().state.value, 42);

        // Count
        let count =
            <SqliteCheckpointer as Checkpointer<TestState>>::count(&checkpointer, "thread-1")
                .await
                .unwrap();
        assert_eq!(count, 1);
    }

    #[tokio::test]
    async fn test_sqlite_checkpointer_list() {
        let checkpointer = SqliteCheckpointer::in_memory().unwrap();

        // Save multiple checkpoints
        for i in 0..5 {
            let state = TestState { value: i };
            let checkpoint = Checkpoint::new("thread-1", state);
            <SqliteCheckpointer as Checkpointer<TestState>>::save(&checkpointer, checkpoint)
                .await
                .unwrap();
        }

        // List all
        let list =
            <SqliteCheckpointer as Checkpointer<TestState>>::list(&checkpointer, "thread-1")
                .await
                .unwrap();
        assert_eq!(list.len(), 5);

        // List paginated
        let paginated =
            <SqliteCheckpointer as Checkpointer<TestState>>::list_paginated(
                &checkpointer,
                "thread-1",
                2,
                1,
            )
            .await
            .unwrap();
        assert_eq!(paginated.len(), 2);
    }

    #[tokio::test]
    async fn test_sqlite_checkpointer_delete() {
        let checkpointer = SqliteCheckpointer::in_memory().unwrap();
        let state = TestState { value: 42 };
        let checkpoint = Checkpoint::new("thread-1", state);

        let id = <SqliteCheckpointer as Checkpointer<TestState>>::save(&checkpointer, checkpoint)
            .await
            .unwrap();

        // Delete checkpoint
        let deleted =
            <SqliteCheckpointer as Checkpointer<TestState>>::delete(&checkpointer, &id)
                .await
                .unwrap();
        assert!(deleted);

        // Verify deleted
        let loaded: Option<Checkpoint<TestState>> =
            <SqliteCheckpointer as Checkpointer<TestState>>::load(&checkpointer, &id)
                .await
                .unwrap();
        assert!(loaded.is_none());
    }

    #[tokio::test]
    async fn test_sqlite_checkpointer_search() {
        let checkpointer = SqliteCheckpointer::in_memory().unwrap();
        let state = TestState { value: 42 };
        let checkpoint = Checkpoint::with_parent("thread-1", "parent-1", state);

        <SqliteCheckpointer as Checkpointer<TestState>>::save(&checkpointer, checkpoint)
            .await
            .unwrap();

        // Search by metadata
        let mut filter = HashMap::new();
        filter.insert("parent_id".to_string(), serde_json::json!("parent-1"));

        let results =
            <SqliteCheckpointer as Checkpointer<TestState>>::search(&checkpointer, "thread-1", filter)
                .await
                .unwrap();
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn test_sqlite_checkpointer_delete_thread() {
        let checkpointer = SqliteCheckpointer::in_memory().unwrap();

        // Create multiple checkpoints in the same thread
        for i in 0..3 {
            let state = TestState { value: i };
            let checkpoint = Checkpoint::new("thread-1", state);
            <SqliteCheckpointer as Checkpointer<TestState>>::save(&checkpointer, checkpoint)
                .await
                .unwrap();
        }

        // Verify they exist
        let count =
            <SqliteCheckpointer as Checkpointer<TestState>>::count(&checkpointer, "thread-1")
                .await
                .unwrap();
        assert_eq!(count, 3);

        // Delete entire thread
        let deleted =
            <SqliteCheckpointer as Checkpointer<TestState>>::delete_thread(&checkpointer, "thread-1")
                .await
                .unwrap();
        assert_eq!(deleted, 3);

        // Verify all deleted
        let count =
            <SqliteCheckpointer as Checkpointer<TestState>>::count(&checkpointer, "thread-1")
                .await
                .unwrap();
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn test_sqlite_embedding_generation() {
        let state = TestState { value: 42 };
        let embedding = SqliteCheckpointer::generate_embedding(&state);

        // Verify embedding properties
        assert_eq!(embedding.len(), 384); // Should be 384 dimensions

        // Verify normalization (magnitude should be ~1.0)
        let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((magnitude - 1.0).abs() < 0.01, "Embedding should be normalized");
    }
}
