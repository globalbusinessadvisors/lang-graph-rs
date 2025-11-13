//! Time travel debugging functionality
//!
//! Provides enterprise-grade execution history tracking, replay capabilities,
//! and step-by-step debugging for graph executions.

use crate::{Result, State};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// A single step in the execution history
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "S: State")]
pub struct ExecutionStep<S: State> {
    /// Step number (0-indexed)
    pub step_number: usize,
    /// Name of the node that was executed
    pub node_name: String,
    /// State before node execution
    pub state_before: S,
    /// State after node execution
    pub state_after: S,
    /// Timestamp of execution
    #[serde(with = "chrono::serde::ts_microseconds")]
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Duration of node execution in microseconds
    pub duration_micros: u64,
    /// Any error that occurred (None if successful)
    pub error: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl<S: State> ExecutionStep<S> {
    /// Create a new execution step
    pub fn new(
        step_number: usize,
        node_name: impl Into<String>,
        state_before: S,
        state_after: S,
        duration_micros: u64,
    ) -> Self {
        Self {
            step_number,
            node_name: node_name.into(),
            state_before,
            state_after,
            timestamp: chrono::Utc::now(),
            duration_micros,
            error: None,
            metadata: HashMap::new(),
        }
    }

    /// Create a step that represents an error
    pub fn with_error(
        step_number: usize,
        node_name: impl Into<String>,
        state_before: S,
        error: impl Into<String>,
    ) -> Self {
        let state_after = state_before.clone();
        Self {
            step_number,
            node_name: node_name.into(),
            state_before,
            state_after,
            timestamp: chrono::Utc::now(),
            duration_micros: 0,
            error: Some(error.into()),
            metadata: HashMap::new(),
        }
    }

    /// Add metadata to the step
    pub fn with_metadata(
        mut self,
        key: impl Into<String>,
        value: serde_json::Value,
    ) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Check if this step resulted in an error
    pub fn is_error(&self) -> bool {
        self.error.is_some()
    }
}

/// Complete execution history for a graph run
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "S: State")]
pub struct ExecutionHistory<S: State> {
    /// Unique history ID
    pub id: String,
    /// Thread/session ID
    pub thread_id: String,
    /// Initial state
    pub initial_state: S,
    /// All execution steps
    pub steps: Vec<ExecutionStep<S>>,
    /// Timestamp when execution started
    #[serde(with = "chrono::serde::ts_seconds")]
    pub started_at: chrono::DateTime<chrono::Utc>,
    /// Timestamp when execution completed (None if still running)
    #[serde(
        with = "chrono::serde::ts_seconds_option",
        skip_serializing_if = "Option::is_none"
    )]
    pub completed_at: Option<chrono::DateTime<chrono::Utc>>,
    /// Whether execution completed successfully
    pub success: bool,
}

impl<S: State> ExecutionHistory<S> {
    /// Create a new execution history
    pub fn new(thread_id: impl Into<String>, initial_state: S) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            thread_id: thread_id.into(),
            initial_state,
            steps: Vec::new(),
            started_at: chrono::Utc::now(),
            completed_at: None,
            success: false,
        }
    }

    /// Add a step to the history
    pub fn add_step(&mut self, step: ExecutionStep<S>) {
        self.steps.push(step);
    }

    /// Mark execution as completed
    pub fn complete(&mut self, success: bool) {
        self.completed_at = Some(chrono::Utc::now());
        self.success = success;
    }

    /// Get the final state (last step's state_after)
    pub fn final_state(&self) -> Option<&S> {
        self.steps.last().map(|s| &s.state_after)
    }

    /// Get state at a specific step
    pub fn state_at_step(&self, step_number: usize) -> Option<&S> {
        if step_number == 0 {
            Some(&self.initial_state)
        } else {
            self.steps.get(step_number - 1).map(|s| &s.state_after)
        }
    }

    /// Get the total number of steps
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    /// Get total execution duration in microseconds
    pub fn total_duration_micros(&self) -> u64 {
        self.steps.iter().map(|s| s.duration_micros).sum()
    }

    /// Get execution path (node names in order)
    pub fn execution_path(&self) -> Vec<String> {
        self.steps.iter().map(|s| s.node_name.clone()).collect()
    }

    /// Check if any step had an error
    pub fn has_errors(&self) -> bool {
        self.steps.iter().any(|s| s.is_error())
    }

    /// Get all steps with errors
    pub fn error_steps(&self) -> Vec<&ExecutionStep<S>> {
        self.steps.iter().filter(|s| s.is_error()).collect()
    }
}

/// Time travel debugger for stepping through execution history
pub struct TimeTravelDebugger<S: State> {
    /// The execution history being debugged
    history: ExecutionHistory<S>,
    /// Current step position (0 = initial state, n = after step n-1)
    current_position: usize,
}

impl<S: State> TimeTravelDebugger<S> {
    /// Create a new debugger from execution history
    pub fn new(history: ExecutionHistory<S>) -> Self {
        Self {
            history,
            current_position: 0,
        }
    }

    /// Get the current state at the debugger's position
    pub fn current_state(&self) -> &S {
        self.history
            .state_at_step(self.current_position)
            .unwrap_or(&self.history.initial_state)
    }

    /// Get the current position
    pub fn position(&self) -> usize {
        self.current_position
    }

    /// Get the maximum position (number of steps)
    pub fn max_position(&self) -> usize {
        self.history.step_count()
    }

    /// Step forward one position
    pub fn step_forward(&mut self) -> Result<&S> {
        if self.current_position < self.max_position() {
            self.current_position += 1;
            Ok(self.current_state())
        } else {
            Err(crate::Error::invalid_operation(
                "Already at the end of execution history",
            ))
        }
    }

    /// Step backward one position
    pub fn step_backward(&mut self) -> Result<&S> {
        if self.current_position > 0 {
            self.current_position -= 1;
            Ok(self.current_state())
        } else {
            Err(crate::Error::invalid_operation(
                "Already at the beginning of execution history",
            ))
        }
    }

    /// Jump to a specific position
    pub fn jump_to(&mut self, position: usize) -> Result<&S> {
        if position <= self.max_position() {
            self.current_position = position;
            Ok(self.current_state())
        } else {
            Err(crate::Error::invalid_operation(format!(
                "Invalid position: {} (max: {})",
                position,
                self.max_position()
            )))
        }
    }

    /// Jump to the beginning
    pub fn jump_to_start(&mut self) -> &S {
        self.current_position = 0;
        self.current_state()
    }

    /// Jump to the end
    pub fn jump_to_end(&mut self) -> &S {
        self.current_position = self.max_position();
        self.current_state()
    }

    /// Get the step at the current position (if any)
    pub fn current_step(&self) -> Option<&ExecutionStep<S>> {
        if self.current_position > 0 {
            self.history.steps.get(self.current_position - 1)
        } else {
            None
        }
    }

    /// Get all steps up to current position
    pub fn steps_up_to_current(&self) -> &[ExecutionStep<S>] {
        &self.history.steps[..self.current_position.min(self.history.step_count())]
    }

    /// Get the execution history
    pub fn history(&self) -> &ExecutionHistory<S> {
        &self.history
    }

    /// Create a diff summary of state changes between two positions
    pub fn diff(&self, from: usize, to: usize) -> Result<StateDiff<S>> {
        if from > self.max_position() || to > self.max_position() {
            return Err(crate::Error::invalid_operation(
                "Position out of bounds for diff",
            ));
        }

        let state_from = self
            .history
            .state_at_step(from)
            .ok_or_else(|| crate::Error::invalid_operation("Invalid from position"))?
            .clone();
        let state_to = self
            .history
            .state_at_step(to)
            .ok_or_else(|| crate::Error::invalid_operation("Invalid to position"))?
            .clone();

        Ok(StateDiff {
            from_position: from,
            to_position: to,
            state_from,
            state_to,
        })
    }
}

/// Represents a difference between two states
#[derive(Debug, Clone, Serialize)]
#[serde(bound = "S: State")]
pub struct StateDiff<S: State> {
    pub from_position: usize,
    pub to_position: usize,
    pub state_from: S,
    pub state_to: S,
}

// Manual Deserialize implementation to avoid type inference issues
impl<'de, S> Deserialize<'de> for StateDiff<S>
where
    S: State,
{
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct StateDiffHelper<S> {
            from_position: usize,
            to_position: usize,
            state_from: S,
            state_to: S,
        }

        let helper = StateDiffHelper::<S>::deserialize(deserializer)?;
        Ok(StateDiff {
            from_position: helper.from_position,
            to_position: helper.to_position,
            state_from: helper.state_from,
            state_to: helper.state_to,
        })
    }
}

/// Manager for tracking execution histories across multiple threads
pub struct ExecutionHistoryManager<S: State> {
    /// Histories indexed by thread ID
    histories: Arc<RwLock<HashMap<String, Vec<ExecutionHistory<S>>>>>,
    /// Currently active histories (one per thread)
    active: Arc<RwLock<HashMap<String, ExecutionHistory<S>>>>,
}

impl<S: State> ExecutionHistoryManager<S> {
    /// Create a new history manager
    pub fn new() -> Self {
        Self {
            histories: Arc::new(RwLock::new(HashMap::new())),
            active: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Start tracking a new execution
    pub async fn start_execution(
        &self,
        thread_id: impl Into<String>,
        initial_state: S,
    ) -> String {
        let thread_id = thread_id.into();
        let history = ExecutionHistory::new(thread_id.clone(), initial_state);
        let history_id = history.id.clone();

        let mut active = self.active.write().await;
        active.insert(thread_id, history);

        history_id
    }

    /// Add a step to the active execution for a thread
    pub async fn add_step(&self, thread_id: &str, step: ExecutionStep<S>) -> Result<()> {
        let mut active = self.active.write().await;

        if let Some(history) = active.get_mut(thread_id) {
            history.add_step(step);
            Ok(())
        } else {
            Err(crate::Error::invalid_operation(format!(
                "No active execution for thread: {}",
                thread_id
            )))
        }
    }

    /// Complete the active execution for a thread
    pub async fn complete_execution(&self, thread_id: &str, success: bool) -> Result<()> {
        let mut active = self.active.write().await;

        if let Some(mut history) = active.remove(thread_id) {
            history.complete(success);

            let mut histories = self.histories.write().await;
            histories
                .entry(thread_id.to_string())
                .or_insert_with(Vec::new)
                .push(history);

            Ok(())
        } else {
            Err(crate::Error::invalid_operation(format!(
                "No active execution for thread: {}",
                thread_id
            )))
        }
    }

    /// Get the active execution history for a thread
    pub async fn get_active_history(&self, thread_id: &str) -> Option<ExecutionHistory<S>> {
        let active = self.active.read().await;
        active.get(thread_id).cloned()
    }

    /// Get all completed histories for a thread
    pub async fn get_histories(&self, thread_id: &str) -> Vec<ExecutionHistory<S>> {
        let histories = self.histories.read().await;
        histories.get(thread_id).cloned().unwrap_or_default()
    }

    /// Get a specific history by ID
    pub async fn get_history(&self, history_id: &str) -> Option<ExecutionHistory<S>> {
        let histories = self.histories.read().await;
        for thread_histories in histories.values() {
            if let Some(history) = thread_histories.iter().find(|h| h.id == history_id) {
                return Some(history.clone());
            }
        }
        None
    }

    /// Create a debugger for a specific history
    pub async fn create_debugger(
        &self,
        history_id: &str,
    ) -> Result<TimeTravelDebugger<S>> {
        let history = self
            .get_history(history_id)
            .await
            .ok_or_else(|| crate::Error::invalid_operation("History not found"))?;

        Ok(TimeTravelDebugger::new(history))
    }

    /// Clear all histories for a thread
    pub async fn clear_thread(&self, thread_id: &str) {
        let mut histories = self.histories.write().await;
        histories.remove(thread_id);

        let mut active = self.active.write().await;
        active.remove(thread_id);
    }

    /// Clear all histories
    pub async fn clear_all(&self) {
        let mut histories = self.histories.write().await;
        histories.clear();

        let mut active = self.active.write().await;
        active.clear();
    }
}

impl<S: State> Default for ExecutionHistoryManager<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: State> Clone for ExecutionHistoryManager<S> {
    fn clone(&self) -> Self {
        Self {
            histories: self.histories.clone(),
            active: self.active.clone(),
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

    #[test]
    fn test_execution_history() {
        let mut history = ExecutionHistory::new("thread-1", TestState { value: 0 });

        let step1 = ExecutionStep::new(
            0,
            "node1",
            TestState { value: 0 },
            TestState { value: 1 },
            100,
        );
        history.add_step(step1);

        let step2 = ExecutionStep::new(
            1,
            "node2",
            TestState { value: 1 },
            TestState { value: 2 },
            150,
        );
        history.add_step(step2);

        assert_eq!(history.step_count(), 2);
        assert_eq!(history.total_duration_micros(), 250);
        assert_eq!(history.final_state().unwrap().value, 2);
        assert_eq!(history.execution_path(), vec!["node1", "node2"]);
    }

    #[test]
    fn test_time_travel_debugger() {
        let mut history = ExecutionHistory::new("thread-1", TestState { value: 0 });

        history.add_step(ExecutionStep::new(
            0,
            "node1",
            TestState { value: 0 },
            TestState { value: 1 },
            100,
        ));
        history.add_step(ExecutionStep::new(
            1,
            "node2",
            TestState { value: 1 },
            TestState { value: 2 },
            150,
        ));
        history.add_step(ExecutionStep::new(
            2,
            "node3",
            TestState { value: 2 },
            TestState { value: 3 },
            200,
        ));

        let mut debugger = TimeTravelDebugger::new(history);

        assert_eq!(debugger.current_state().value, 0);
        assert_eq!(debugger.position(), 0);

        debugger.step_forward().unwrap();
        assert_eq!(debugger.current_state().value, 1);

        debugger.step_forward().unwrap();
        assert_eq!(debugger.current_state().value, 2);

        debugger.step_backward().unwrap();
        assert_eq!(debugger.current_state().value, 1);

        debugger.jump_to_end();
        assert_eq!(debugger.current_state().value, 3);

        debugger.jump_to_start();
        assert_eq!(debugger.current_state().value, 0);
    }

    #[tokio::test]
    async fn test_history_manager() {
        let manager = ExecutionHistoryManager::<TestState>::new();

        let history_id = manager
            .start_execution("thread-1", TestState { value: 0 })
            .await;

        manager
            .add_step(
                "thread-1",
                ExecutionStep::new(
                    0,
                    "node1",
                    TestState { value: 0 },
                    TestState { value: 1 },
                    100,
                ),
            )
            .await
            .unwrap();

        let active = manager.get_active_history("thread-1").await.unwrap();
        assert_eq!(active.step_count(), 1);

        manager.complete_execution("thread-1", true).await.unwrap();

        let histories = manager.get_histories("thread-1").await;
        assert_eq!(histories.len(), 1);
        assert!(histories[0].success);
    }
}
