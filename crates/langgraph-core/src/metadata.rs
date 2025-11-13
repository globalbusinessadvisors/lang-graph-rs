//! Execution metadata tracking for time travel debugging

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Metadata for a single node execution
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NodeExecutionMetadata {
    /// Node name
    pub node_name: String,
    /// Step number in execution
    pub step: usize,
    /// Start timestamp
    pub started_at: DateTime<Utc>,
    /// End timestamp
    pub completed_at: Option<DateTime<Utc>>,
    /// Execution duration
    pub duration_ms: Option<u128>,
    /// Whether execution succeeded
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
    /// Custom metadata
    pub extra: HashMap<String, serde_json::Value>,
}

impl NodeExecutionMetadata {
    /// Create new node execution metadata
    pub fn new(node_name: impl Into<String>, step: usize) -> Self {
        Self {
            node_name: node_name.into(),
            step,
            started_at: Utc::now(),
            completed_at: None,
            duration_ms: None,
            success: false,
            error: None,
            extra: HashMap::new(),
        }
    }

    /// Mark execution as completed successfully
    pub fn complete(mut self) -> Self {
        let now = Utc::now();
        let duration = now.signed_duration_since(self.started_at);
        self.completed_at = Some(now);
        self.duration_ms = Some(duration.num_milliseconds() as u128);
        self.success = true;
        self
    }

    /// Mark execution as failed
    pub fn fail(mut self, error: impl Into<String>) -> Self {
        let now = Utc::now();
        let duration = now.signed_duration_since(self.started_at);
        self.completed_at = Some(now);
        self.duration_ms = Some(duration.num_milliseconds() as u128);
        self.success = false;
        self.error = Some(error.into());
        self
    }

    /// Add custom metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.extra.insert(key.into(), value);
        self
    }

    /// Get execution duration as Duration
    pub fn duration(&self) -> Option<Duration> {
        self.duration_ms.map(|ms| Duration::from_millis(ms as u64))
    }
}

/// Decision made during conditional edge evaluation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConditionalDecision {
    /// From node
    pub from_node: String,
    /// To node (chosen destination)
    pub to_node: String,
    /// Condition key that was evaluated
    pub condition_key: String,
    /// Step number
    pub step: usize,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// All available options in the edge map
    pub available_options: Vec<String>,
}

impl ConditionalDecision {
    /// Create new conditional decision record
    pub fn new(
        from_node: impl Into<String>,
        to_node: impl Into<String>,
        condition_key: impl Into<String>,
        step: usize,
        available_options: Vec<String>,
    ) -> Self {
        Self {
            from_node: from_node.into(),
            to_node: to_node.into(),
            condition_key: condition_key.into(),
            step,
            timestamp: Utc::now(),
            available_options,
        }
    }
}

/// Complete execution trace for a graph run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionTrace {
    /// Unique trace ID
    pub trace_id: String,
    /// Thread ID
    pub thread_id: String,
    /// Start timestamp
    pub started_at: DateTime<Utc>,
    /// End timestamp
    pub completed_at: Option<DateTime<Utc>>,
    /// Node execution history
    pub node_executions: Vec<NodeExecutionMetadata>,
    /// Conditional decisions made
    pub decisions: Vec<ConditionalDecision>,
    /// Entry point
    pub entry_point: String,
    /// Exit point (if reached)
    pub exit_point: Option<String>,
    /// Whether execution completed successfully
    pub success: bool,
    /// Final error if any
    pub error: Option<String>,
    /// Custom trace metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl ExecutionTrace {
    /// Create new execution trace
    pub fn new(trace_id: impl Into<String>, thread_id: impl Into<String>, entry_point: impl Into<String>) -> Self {
        Self {
            trace_id: trace_id.into(),
            thread_id: thread_id.into(),
            started_at: Utc::now(),
            completed_at: None,
            node_executions: Vec::new(),
            decisions: Vec::new(),
            entry_point: entry_point.into(),
            exit_point: None,
            success: false,
            error: None,
            metadata: HashMap::new(),
        }
    }

    /// Add node execution metadata
    pub fn add_node_execution(&mut self, metadata: NodeExecutionMetadata) {
        self.node_executions.push(metadata);
    }

    /// Add conditional decision
    pub fn add_decision(&mut self, decision: ConditionalDecision) {
        self.decisions.push(decision);
    }

    /// Mark trace as completed
    pub fn complete(&mut self, exit_point: impl Into<String>) {
        self.completed_at = Some(Utc::now());
        self.exit_point = Some(exit_point.into());
        self.success = true;
    }

    /// Mark trace as failed
    pub fn fail(&mut self, error: impl Into<String>) {
        self.completed_at = Some(Utc::now());
        self.success = false;
        self.error = Some(error.into());
    }

    /// Add custom metadata
    pub fn add_metadata(&mut self, key: impl Into<String>, value: serde_json::Value) {
        self.metadata.insert(key.into(), value);
    }

    /// Get total execution duration
    pub fn total_duration_ms(&self) -> Option<u128> {
        self.completed_at.map(|completed| {
            let duration = completed.signed_duration_since(self.started_at);
            duration.num_milliseconds() as u128
        })
    }

    /// Get execution path (sequence of nodes)
    pub fn execution_path(&self) -> Vec<String> {
        self.node_executions.iter().map(|e| e.node_name.clone()).collect()
    }

    /// Get failed nodes
    pub fn failed_nodes(&self) -> Vec<&NodeExecutionMetadata> {
        self.node_executions.iter().filter(|e| !e.success).collect()
    }

    /// Get successful nodes
    pub fn successful_nodes(&self) -> Vec<&NodeExecutionMetadata> {
        self.node_executions.iter().filter(|e| e.success).collect()
    }

    /// Get total number of steps
    pub fn total_steps(&self) -> usize {
        self.node_executions.len()
    }

    /// Get average node execution time
    pub fn average_node_duration_ms(&self) -> Option<u128> {
        let durations: Vec<u128> = self.node_executions.iter().filter_map(|e| e.duration_ms).collect();
        if durations.is_empty() {
            None
        } else {
            Some(durations.iter().sum::<u128>() / durations.len() as u128)
        }
    }
}

/// Builder for execution timeline visualization data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionTimeline {
    /// Timeline entries
    pub entries: Vec<TimelineEntry>,
    /// Start time
    pub start_time: DateTime<Utc>,
    /// End time
    pub end_time: Option<DateTime<Utc>>,
}

/// Single entry in execution timeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineEntry {
    /// Event type
    pub event_type: TimelineEventType,
    /// Node name
    pub node_name: String,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Duration if applicable
    pub duration_ms: Option<u128>,
    /// Additional details
    pub details: HashMap<String, serde_json::Value>,
}

/// Timeline event types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum TimelineEventType {
    /// Node started
    NodeStart,
    /// Node completed
    NodeComplete,
    /// Node failed
    NodeError,
    /// Conditional evaluated
    ConditionalEvaluated,
    /// Checkpoint created
    CheckpointCreated,
    /// Execution paused
    Paused,
    /// Execution resumed
    Resumed,
}

impl ExecutionTimeline {
    /// Create new timeline
    pub fn new(start_time: DateTime<Utc>) -> Self {
        Self {
            entries: Vec::new(),
            start_time,
            end_time: None,
        }
    }

    /// Add timeline entry
    pub fn add_entry(&mut self, entry: TimelineEntry) {
        self.entries.push(entry);
    }

    /// Complete timeline
    pub fn complete(&mut self) {
        self.end_time = Some(Utc::now());
    }

    /// Get entries for a specific node
    pub fn entries_for_node(&self, node_name: &str) -> Vec<&TimelineEntry> {
        self.entries.iter().filter(|e| e.node_name == node_name).collect()
    }

    /// Get entries of a specific type
    pub fn entries_of_type(&self, event_type: TimelineEventType) -> Vec<&TimelineEntry> {
        self.entries.iter().filter(|e| e.event_type == event_type).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_execution_metadata() {
        let metadata = NodeExecutionMetadata::new("test_node", 1);
        assert_eq!(metadata.node_name, "test_node");
        assert_eq!(metadata.step, 1);
        assert!(!metadata.success);
    }

    #[test]
    fn test_node_execution_complete() {
        let metadata = NodeExecutionMetadata::new("test_node", 1).complete();
        assert!(metadata.success);
        assert!(metadata.completed_at.is_some());
        assert!(metadata.duration_ms.is_some());
        assert!(metadata.error.is_none());
    }

    #[test]
    fn test_node_execution_fail() {
        let metadata = NodeExecutionMetadata::new("test_node", 1).fail("Test error");
        assert!(!metadata.success);
        assert!(metadata.completed_at.is_some());
        assert!(metadata.duration_ms.is_some());
        assert_eq!(metadata.error, Some("Test error".to_string()));
    }

    #[test]
    fn test_execution_trace() {
        let mut trace = ExecutionTrace::new("trace-1", "thread-1", "start");
        assert_eq!(trace.trace_id, "trace-1");
        assert_eq!(trace.thread_id, "thread-1");
        assert_eq!(trace.entry_point, "start");
        assert_eq!(trace.total_steps(), 0);

        let metadata = NodeExecutionMetadata::new("node1", 1).complete();
        trace.add_node_execution(metadata);
        assert_eq!(trace.total_steps(), 1);

        trace.complete("end");
        assert!(trace.success);
        assert_eq!(trace.exit_point, Some("end".to_string()));
    }

    #[test]
    fn test_execution_path() {
        let mut trace = ExecutionTrace::new("trace-1", "thread-1", "start");
        trace.add_node_execution(NodeExecutionMetadata::new("node1", 1).complete());
        trace.add_node_execution(NodeExecutionMetadata::new("node2", 2).complete());
        trace.add_node_execution(NodeExecutionMetadata::new("node3", 3).complete());

        let path = trace.execution_path();
        assert_eq!(path, vec!["node1", "node2", "node3"]);
    }

    #[test]
    fn test_conditional_decision() {
        let decision = ConditionalDecision::new(
            "node1",
            "node2",
            "condition_true",
            1,
            vec!["node2".to_string(), "node3".to_string()],
        );

        assert_eq!(decision.from_node, "node1");
        assert_eq!(decision.to_node, "node2");
        assert_eq!(decision.condition_key, "condition_true");
        assert_eq!(decision.available_options.len(), 2);
    }

    #[test]
    fn test_execution_timeline() {
        let mut timeline = ExecutionTimeline::new(Utc::now());
        timeline.add_entry(TimelineEntry {
            event_type: TimelineEventType::NodeStart,
            node_name: "node1".to_string(),
            timestamp: Utc::now(),
            duration_ms: None,
            details: HashMap::new(),
        });

        assert_eq!(timeline.entries.len(), 1);
        let node_entries = timeline.entries_for_node("node1");
        assert_eq!(node_entries.len(), 1);
    }
}
