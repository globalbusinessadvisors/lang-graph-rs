//! Enhanced streaming execution with events and configuration

use crate::{Result, State};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Configuration for streaming execution
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Maximum number of steps to execute
    pub max_steps: Option<usize>,
    /// Enable checkpoint emission after each step
    pub emit_checkpoints: bool,
    /// Enable execution metadata tracking
    pub track_metadata: bool,
    /// Filter which nodes emit events (None = all nodes)
    pub node_filter: Option<Vec<String>>,
    /// Batch size for buffering events (1 = no buffering)
    pub batch_size: usize,
    /// Timeout for each node execution
    pub node_timeout: Option<Duration>,
    /// Enable debug logging
    pub debug: bool,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            max_steps: Some(1000),
            emit_checkpoints: true,
            track_metadata: true,
            node_filter: None,
            batch_size: 1,
            node_timeout: Some(Duration::from_secs(300)), // 5 minutes default
            debug: false,
        }
    }
}

impl StreamConfig {
    /// Create a new stream config with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum steps
    pub fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = Some(max_steps);
        self
    }

    /// Enable checkpoint emission
    pub fn with_checkpoints(mut self, enable: bool) -> Self {
        self.emit_checkpoints = enable;
        self
    }

    /// Enable metadata tracking
    pub fn with_metadata(mut self, enable: bool) -> Self {
        self.track_metadata = enable;
        self
    }

    /// Filter nodes that emit events
    pub fn with_node_filter(mut self, nodes: Vec<String>) -> Self {
        self.node_filter = Some(nodes);
        self
    }

    /// Set batch size for event buffering
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size.max(1);
        self
    }

    /// Set node execution timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.node_timeout = Some(timeout);
        self
    }

    /// Enable debug mode
    pub fn with_debug(mut self, enable: bool) -> Self {
        self.debug = enable;
        self
    }

    /// Check if a node should emit events
    pub fn should_emit_node(&self, node_name: &str) -> bool {
        match &self.node_filter {
            Some(filter) => filter.iter().any(|n| n == node_name),
            None => true,
        }
    }
}

/// Events emitted during streaming execution
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", bound = "S: State")]
pub enum StreamEvent<S: State> {
    /// Execution started
    Started {
        /// Entry point node
        entry_node: String,
        /// Start timestamp
        timestamp: DateTime<Utc>,
    },

    /// Node execution started
    NodeStart {
        /// Node name
        node_name: String,
        /// Start timestamp
        timestamp: DateTime<Utc>,
        /// Step number
        step: usize,
    },

    /// Node execution completed
    NodeComplete {
        /// Node name
        node_name: String,
        /// Updated state after execution
        state: S,
        /// Completion timestamp
        timestamp: DateTime<Utc>,
        /// Step number
        step: usize,
        /// Execution duration
        duration_ms: u128,
    },

    /// Node execution failed
    NodeError {
        /// Node name
        node_name: String,
        /// Error message
        error: String,
        /// Error timestamp
        timestamp: DateTime<Utc>,
        /// Step number
        step: usize,
    },

    /// Conditional edge evaluated
    ConditionalEdge {
        /// From node
        from_node: String,
        /// Selected destination
        to_node: String,
        /// Condition result key
        condition_key: String,
        /// Timestamp
        timestamp: DateTime<Utc>,
    },

    /// Checkpoint created
    CheckpointCreated {
        /// Checkpoint ID
        checkpoint_id: String,
        /// Node where checkpoint was created
        node_name: String,
        /// Thread ID
        thread_id: String,
        /// Timestamp
        timestamp: DateTime<Utc>,
    },

    /// Execution paused (for human-in-the-loop)
    Paused {
        /// Node where execution paused
        node_name: String,
        /// Reason for pause
        reason: String,
        /// Checkpoint ID for resumption
        checkpoint_id: Option<String>,
        /// Timestamp
        timestamp: DateTime<Utc>,
    },

    /// Execution resumed
    Resumed {
        /// Node where execution resumed
        node_name: String,
        /// Checkpoint ID resumed from
        checkpoint_id: Option<String>,
        /// Timestamp
        timestamp: DateTime<Utc>,
    },

    /// Execution completed
    Completed {
        /// Final node
        final_node: String,
        /// Final state
        state: S,
        /// Completion timestamp
        timestamp: DateTime<Utc>,
        /// Total steps executed
        total_steps: usize,
        /// Total execution duration
        total_duration_ms: u128,
    },

    /// Execution failed
    Failed {
        /// Node where failure occurred
        node_name: String,
        /// Error message
        error: String,
        /// Timestamp
        timestamp: DateTime<Utc>,
    },
}

impl<S: State> StreamEvent<S> {
    /// Get the timestamp of this event
    pub fn timestamp(&self) -> &DateTime<Utc> {
        match self {
            StreamEvent::Started { timestamp, .. }
            | StreamEvent::NodeStart { timestamp, .. }
            | StreamEvent::NodeComplete { timestamp, .. }
            | StreamEvent::NodeError { timestamp, .. }
            | StreamEvent::ConditionalEdge { timestamp, .. }
            | StreamEvent::CheckpointCreated { timestamp, .. }
            | StreamEvent::Paused { timestamp, .. }
            | StreamEvent::Resumed { timestamp, .. }
            | StreamEvent::Completed { timestamp, .. }
            | StreamEvent::Failed { timestamp, .. } => timestamp,
        }
    }

    /// Get the associated node name if applicable
    pub fn node_name(&self) -> Option<&str> {
        match self {
            StreamEvent::Started { entry_node, .. } => Some(entry_node),
            StreamEvent::NodeStart { node_name, .. }
            | StreamEvent::NodeComplete { node_name, .. }
            | StreamEvent::NodeError { node_name, .. }
            | StreamEvent::CheckpointCreated { node_name, .. }
            | StreamEvent::Paused { node_name, .. }
            | StreamEvent::Resumed { node_name, .. }
            | StreamEvent::Failed { node_name, .. } => Some(node_name),
            StreamEvent::ConditionalEdge { from_node, .. } => Some(from_node),
            StreamEvent::Completed { final_node, .. } => Some(final_node),
        }
    }

    /// Check if this is an error event
    pub fn is_error(&self) -> bool {
        matches!(self, StreamEvent::NodeError { .. } | StreamEvent::Failed { .. })
    }

    /// Check if this is a completion event
    pub fn is_complete(&self) -> bool {
        matches!(self, StreamEvent::Completed { .. })
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
    fn test_stream_config_defaults() {
        let config = StreamConfig::default();
        assert_eq!(config.max_steps, Some(1000));
        assert!(config.emit_checkpoints);
        assert!(config.track_metadata);
        assert_eq!(config.batch_size, 1);
    }

    #[test]
    fn test_stream_config_builder() {
        let config = StreamConfig::new()
            .with_max_steps(100)
            .with_checkpoints(false)
            .with_batch_size(5);

        assert_eq!(config.max_steps, Some(100));
        assert!(!config.emit_checkpoints);
        assert_eq!(config.batch_size, 5);
    }

    #[test]
    fn test_stream_config_node_filter() {
        let config = StreamConfig::new()
            .with_node_filter(vec!["node1".to_string(), "node2".to_string()]);

        assert!(config.should_emit_node("node1"));
        assert!(config.should_emit_node("node2"));
        assert!(!config.should_emit_node("node3"));
    }

    #[test]
    fn test_stream_event_timestamp() {
        let now = Utc::now();
        let event = StreamEvent::<TestState>::Started {
            entry_node: "start".to_string(),
            timestamp: now,
        };

        assert_eq!(event.timestamp(), &now);
    }

    #[test]
    fn test_stream_event_node_name() {
        let event = StreamEvent::<TestState>::NodeStart {
            node_name: "test_node".to_string(),
            timestamp: Utc::now(),
            step: 1,
        };

        assert_eq!(event.node_name(), Some("test_node"));
    }

    #[test]
    fn test_stream_event_is_error() {
        let error_event = StreamEvent::<TestState>::NodeError {
            node_name: "test".to_string(),
            error: "test error".to_string(),
            timestamp: Utc::now(),
            step: 1,
        };

        assert!(error_event.is_error());

        let complete_event = StreamEvent::<TestState>::NodeComplete {
            node_name: "test".to_string(),
            state: TestState { value: 42 },
            timestamp: Utc::now(),
            step: 1,
            duration_ms: 100,
        };

        assert!(!complete_event.is_error());
    }
}
