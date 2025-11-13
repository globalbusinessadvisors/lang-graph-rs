//! Time travel debugging with replay and state diffing
//!
//! This module provides utilities for time travel debugging. To use with checkpoints,
//! you'll need to also depend on the `langgraph-checkpoint` crate which provides
//! the `Checkpointer` trait and `Checkpoint` type.

use crate::{Error, Result, State};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::HashMap;

/// State difference between two checkpoints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateDiff {
    /// From checkpoint ID
    pub from_checkpoint: String,
    /// To checkpoint ID
    pub to_checkpoint: String,
    /// Fields that were added
    pub added: Vec<FieldChange>,
    /// Fields that were modified
    pub modified: Vec<FieldChange>,
    /// Fields that were deleted
    pub deleted: Vec<FieldChange>,
}

/// Individual field change in state
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FieldChange {
    /// JSON path to the field (e.g., "user.name")
    pub path: String,
    /// Old value (None for additions)
    pub old_value: Option<JsonValue>,
    /// New value (None for deletions)
    pub new_value: Option<JsonValue>,
}

impl FieldChange {
    /// Create a field addition
    pub fn added(path: impl Into<String>, new_value: JsonValue) -> Self {
        Self {
            path: path.into(),
            old_value: None,
            new_value: Some(new_value),
        }
    }

    /// Create a field modification
    pub fn modified(path: impl Into<String>, old_value: JsonValue, new_value: JsonValue) -> Self {
        Self {
            path: path.into(),
            old_value: Some(old_value),
            new_value: Some(new_value),
        }
    }

    /// Create a field deletion
    pub fn deleted(path: impl Into<String>, old_value: JsonValue) -> Self {
        Self {
            path: path.into(),
            old_value: Some(old_value),
            new_value: None,
        }
    }
}

impl StateDiff {
    /// Create new state diff
    pub fn new(from_checkpoint: impl Into<String>, to_checkpoint: impl Into<String>) -> Self {
        Self {
            from_checkpoint: from_checkpoint.into(),
            to_checkpoint: to_checkpoint.into(),
            added: Vec::new(),
            modified: Vec::new(),
            deleted: Vec::new(),
        }
    }

    /// Add a field addition
    pub fn add_addition(&mut self, change: FieldChange) {
        self.added.push(change);
    }

    /// Add a field modification
    pub fn add_modification(&mut self, change: FieldChange) {
        self.modified.push(change);
    }

    /// Add a field deletion
    pub fn add_deletion(&mut self, change: FieldChange) {
        self.deleted.push(change);
    }

    /// Check if there are any differences
    pub fn has_changes(&self) -> bool {
        !self.added.is_empty() || !self.modified.is_empty() || !self.deleted.is_empty()
    }

    /// Get total number of changes
    pub fn total_changes(&self) -> usize {
        self.added.len() + self.modified.len() + self.deleted.len()
    }
}

/// State differ for computing differences
pub struct StateDiffer;

impl StateDiffer {
    /// Compute difference between two states
    pub fn diff<S: State>(
        from_checkpoint: impl Into<String>,
        to_checkpoint: impl Into<String>,
        from_state: &S,
        to_state: &S,
    ) -> Result<StateDiff> {
        let from_json = serde_json::to_value(from_state)
            .map_err(|e| Error::serialization_error(format!("Failed to serialize from_state: {}", e)))?;
        let to_json = serde_json::to_value(to_state)
            .map_err(|e| Error::serialization_error(format!("Failed to serialize to_state: {}", e)))?;

        let mut diff = StateDiff::new(from_checkpoint, to_checkpoint);
        Self::diff_values("", &from_json, &to_json, &mut diff);

        Ok(diff)
    }

    /// Recursively diff JSON values
    fn diff_values(path: &str, from: &JsonValue, to: &JsonValue, diff: &mut StateDiff) {
        match (from, to) {
            (JsonValue::Object(from_obj), JsonValue::Object(to_obj)) => {
                // Check for modifications and deletions
                for (key, from_val) in from_obj {
                    let field_path = if path.is_empty() {
                        key.clone()
                    } else {
                        format!("{}.{}", path, key)
                    };

                    match to_obj.get(key) {
                        Some(to_val) => {
                            if from_val != to_val {
                                if Self::is_primitive(from_val) && Self::is_primitive(to_val) {
                                    diff.add_modification(FieldChange::modified(
                                        field_path,
                                        from_val.clone(),
                                        to_val.clone(),
                                    ));
                                } else {
                                    Self::diff_values(&field_path, from_val, to_val, diff);
                                }
                            }
                        }
                        None => {
                            diff.add_deletion(FieldChange::deleted(field_path, from_val.clone()));
                        }
                    }
                }

                // Check for additions
                for (key, to_val) in to_obj {
                    if !from_obj.contains_key(key) {
                        let field_path = if path.is_empty() {
                            key.clone()
                        } else {
                            format!("{}.{}", path, key)
                        };
                        diff.add_addition(FieldChange::added(field_path, to_val.clone()));
                    }
                }
            }
            (JsonValue::Array(from_arr), JsonValue::Array(to_arr)) => {
                // For arrays, compare element by element
                let max_len = from_arr.len().max(to_arr.len());
                for i in 0..max_len {
                    let field_path = format!("{}[{}]", path, i);
                    match (from_arr.get(i), to_arr.get(i)) {
                        (Some(from_val), Some(to_val)) => {
                            if from_val != to_val {
                                if Self::is_primitive(from_val) && Self::is_primitive(to_val) {
                                    diff.add_modification(FieldChange::modified(
                                        field_path,
                                        from_val.clone(),
                                        to_val.clone(),
                                    ));
                                } else {
                                    Self::diff_values(&field_path, from_val, to_val, diff);
                                }
                            }
                        }
                        (Some(from_val), None) => {
                            diff.add_deletion(FieldChange::deleted(field_path, from_val.clone()));
                        }
                        (None, Some(to_val)) => {
                            diff.add_addition(FieldChange::added(field_path, to_val.clone()));
                        }
                        (None, None) => {}
                    }
                }
            }
            _ => {
                // Primitive values or type mismatch
                if from != to && !path.is_empty() {
                    diff.add_modification(FieldChange::modified(path, from.clone(), to.clone()));
                }
            }
        }
    }

    /// Check if a value is primitive (not object or array)
    fn is_primitive(value: &JsonValue) -> bool {
        !matches!(value, JsonValue::Object(_) | JsonValue::Array(_))
    }
}

/// Replay configuration
#[derive(Debug, Clone)]
pub struct ReplayConfig {
    /// Stop replay at specific node
    pub stop_at_node: Option<String>,
    /// Maximum steps to replay
    pub max_steps: Option<usize>,
    /// Enable debug logging
    pub debug: bool,
    /// Emit events during replay
    pub emit_events: bool,
}

impl Default for ReplayConfig {
    fn default() -> Self {
        Self {
            stop_at_node: None,
            max_steps: Some(1000),
            debug: false,
            emit_events: true,
        }
    }
}

impl ReplayConfig {
    /// Create new replay config
    pub fn new() -> Self {
        Self::default()
    }

    /// Set stop node
    pub fn stop_at(mut self, node_name: impl Into<String>) -> Self {
        self.stop_at_node = Some(node_name.into());
        self
    }

    /// Set max steps
    pub fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = Some(max_steps);
        self
    }

    /// Enable debug mode
    pub fn with_debug(mut self, enable: bool) -> Self {
        self.debug = enable;
        self
    }

    /// Enable event emission
    pub fn with_events(mut self, enable: bool) -> Self {
        self.emit_events = enable;
        self
    }
}

/// Checkpoint comparison result
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "S: State")]
pub struct CheckpointComparison<S: State> {
    /// Checkpoint metadata for first checkpoint
    pub first_checkpoint_id: String,
    /// Checkpoint metadata for second checkpoint
    pub second_checkpoint_id: String,
    /// State difference
    pub diff: StateDiff,
    /// First state
    pub first_state: S,
    /// Second state
    pub second_state: S,
}

impl<S: State> CheckpointComparison<S> {
    /// Create new checkpoint comparison
    pub fn new(
        first_checkpoint_id: impl Into<String>,
        second_checkpoint_id: impl Into<String>,
        first_state: S,
        second_state: S,
        diff: StateDiff,
    ) -> Self {
        Self {
            first_checkpoint_id: first_checkpoint_id.into(),
            second_checkpoint_id: second_checkpoint_id.into(),
            diff,
            first_state,
            second_state,
        }
    }
}

/// Minimal checkpoint interface for time travel operations
///
/// This trait provides the essential operations needed for time travel debugging.
/// It's designed to be compatible with `langgraph-checkpoint::Checkpointer`.
#[async_trait]
pub trait CheckpointLoader<S: State>: Send + Sync {
    /// Checkpoint metadata type
    type Metadata: CheckpointMetadata;

    /// Load a checkpoint by ID
    async fn load_checkpoint(&self, checkpoint_id: &str) -> Result<Option<CheckpointData<S>>>;

    /// List checkpoint metadata for a thread
    async fn list_metadata(&self, thread_id: &str) -> Result<Vec<Self::Metadata>>;
}

/// Checkpoint metadata trait
pub trait CheckpointMetadata: Send + Sync {
    /// Get checkpoint ID
    fn id(&self) -> &str;

    /// Get creation timestamp
    fn created_at(&self) -> DateTime<Utc>;
}

/// Checkpoint data containing state
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "S: State")]
pub struct CheckpointData<S: State> {
    /// Checkpoint ID
    pub id: String,
    /// State at this checkpoint
    pub state: S,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
}

impl<S: State> CheckpointData<S> {
    /// Create new checkpoint data
    pub fn new(id: impl Into<String>, state: S, created_at: DateTime<Utc>) -> Self {
        Self {
            id: id.into(),
            state,
            created_at,
        }
    }
}

/// Time travel operations for checkpoint debugging
pub struct TimeTravelDebugger;

impl TimeTravelDebugger {
    /// Compare two checkpoints
    pub async fn compare_checkpoints<S: State, C: CheckpointLoader<S>>(
        checkpointer: &C,
        first_id: &str,
        second_id: &str,
    ) -> Result<CheckpointComparison<S>> {
        let first = checkpointer
            .load_checkpoint(first_id)
            .await?
            .ok_or_else(|| Error::checkpoint_error(format!("Checkpoint not found: {}", first_id)))?;

        let second = checkpointer
            .load_checkpoint(second_id)
            .await?
            .ok_or_else(|| Error::checkpoint_error(format!("Checkpoint not found: {}", second_id)))?;

        let diff = StateDiffer::diff(first_id, second_id, &first.state, &second.state)?;

        Ok(CheckpointComparison::new(
            first_id,
            second_id,
            first.state,
            second.state,
            diff,
        ))
    }

    /// Get checkpoint history for a thread
    pub async fn get_history<S: State, C: CheckpointLoader<S>>(
        checkpointer: &C,
        thread_id: &str,
    ) -> Result<Vec<CheckpointData<S>>> {
        let metadata_list = checkpointer.list_metadata(thread_id).await?;
        let mut checkpoints = Vec::new();

        for metadata in metadata_list {
            if let Some(checkpoint) = checkpointer.load_checkpoint(metadata.id()).await? {
                checkpoints.push(checkpoint);
            }
        }

        Ok(checkpoints)
    }

    /// Find checkpoint by criteria
    pub async fn find_checkpoint<S: State, C: CheckpointLoader<S>>(
        checkpointer: &C,
        thread_id: &str,
        predicate: impl Fn(&CheckpointData<S>) -> bool,
    ) -> Result<Option<CheckpointData<S>>> {
        let history = Self::get_history(checkpointer, thread_id).await?;
        Ok(history.into_iter().find(predicate))
    }

    /// Get state at specific checkpoint
    pub async fn get_state_at<S: State, C: CheckpointLoader<S>>(
        checkpointer: &C,
        checkpoint_id: &str,
    ) -> Result<S> {
        let checkpoint = checkpointer
            .load_checkpoint(checkpoint_id)
            .await?
            .ok_or_else(|| Error::checkpoint_error(format!("Checkpoint not found: {}", checkpoint_id)))?;

        Ok(checkpoint.state)
    }

    /// Build execution timeline from checkpoints
    pub async fn build_timeline<S: State, C: CheckpointLoader<S>>(
        checkpointer: &C,
        thread_id: &str,
    ) -> Result<Vec<TimelinePoint<S>>> {
        let history = Self::get_history(checkpointer, thread_id).await?;
        let mut timeline = Vec::new();

        for (index, checkpoint) in history.iter().enumerate() {
            let diff = if index > 0 {
                Some(StateDiffer::diff(
                    &history[index - 1].id,
                    &checkpoint.id,
                    &history[index - 1].state,
                    &checkpoint.state,
                )?)
            } else {
                None
            };

            timeline.push(TimelinePoint {
                checkpoint_id: checkpoint.id.clone(),
                timestamp: checkpoint.created_at,
                state: checkpoint.state.clone(),
                diff,
                step: index + 1,
            });
        }

        Ok(timeline)
    }
}

/// Point in execution timeline
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "S: State")]
pub struct TimelinePoint<S: State> {
    /// Checkpoint ID
    pub checkpoint_id: String,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// State at this point
    pub state: S,
    /// Diff from previous point
    pub diff: Option<StateDiff>,
    /// Step number
    pub step: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct TestState {
        name: String,
        count: i32,
        tags: Vec<String>,
    }

    #[test]
    fn test_field_change() {
        let change = FieldChange::modified("name", JsonValue::String("old".to_string()), JsonValue::String("new".to_string()));
        assert_eq!(change.path, "name");
        assert_eq!(change.old_value, Some(JsonValue::String("old".to_string())));
        assert_eq!(change.new_value, Some(JsonValue::String("new".to_string())));
    }

    #[test]
    fn test_state_diff_simple() {
        let state1 = TestState {
            name: "Alice".to_string(),
            count: 5,
            tags: vec!["a".to_string()],
        };

        let state2 = TestState {
            name: "Bob".to_string(),
            count: 5,
            tags: vec!["a".to_string()],
        };

        let diff = StateDiffer::diff("cp1", "cp2", &state1, &state2).unwrap();
        assert!(diff.has_changes());
        assert_eq!(diff.modified.len(), 1);
        assert_eq!(diff.modified[0].path, "name");
    }

    #[test]
    fn test_state_diff_complex() {
        let state1 = TestState {
            name: "Alice".to_string(),
            count: 5,
            tags: vec!["a".to_string(), "b".to_string()],
        };

        let state2 = TestState {
            name: "Alice".to_string(),
            count: 10,
            tags: vec!["a".to_string()],
        };

        let diff = StateDiffer::diff("cp1", "cp2", &state1, &state2).unwrap();
        assert!(diff.has_changes());

        // Should detect count change and tags array change
        assert!(diff.total_changes() >= 1);
    }

    #[test]
    fn test_replay_config() {
        let config = ReplayConfig::new()
            .stop_at("node3")
            .with_max_steps(100)
            .with_debug(true);

        assert_eq!(config.stop_at_node, Some("node3".to_string()));
        assert_eq!(config.max_steps, Some(100));
        assert!(config.debug);
    }

    #[test]
    fn test_state_diff_no_changes() {
        let state1 = TestState {
            name: "Alice".to_string(),
            count: 5,
            tags: vec!["a".to_string()],
        };

        let state2 = state1.clone();

        let diff = StateDiffer::diff("cp1", "cp2", &state1, &state2).unwrap();
        assert!(!diff.has_changes());
        assert_eq!(diff.total_changes(), 0);
    }
}
