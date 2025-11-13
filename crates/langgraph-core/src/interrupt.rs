//! Human-in-the-loop interrupt and pause/resume mechanism

use crate::{Error, Result, State};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};

/// Interrupt signal for pausing execution
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InterruptSignal {
    /// Pause execution before node
    PauseBefore {
        /// Node to pause before
        node_name: String,
        /// Reason for pause
        reason: String,
    },

    /// Pause execution after node
    PauseAfter {
        /// Node to pause after
        node_name: String,
        /// Reason for pause
        reason: String,
    },

    /// Request approval to continue
    ApprovalRequired {
        /// Node requiring approval
        node_name: String,
        /// Approval prompt/question
        prompt: String,
        /// Approval options
        options: Vec<String>,
    },

    /// Cancel execution
    Cancel {
        /// Reason for cancellation
        reason: String,
    },

    /// Resume execution
    Resume,
}

/// Response to an interrupt signal
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InterruptResponse {
    /// Approved to continue
    Approved {
        /// Selected option (if applicable)
        selected_option: Option<String>,
        /// Additional data from user
        data: HashMap<String, serde_json::Value>,
    },

    /// Rejected/denied
    Rejected {
        /// Reason for rejection
        reason: String,
    },

    /// State modification requested
    ModifyState {
        /// State modifications as JSON patches
        modifications: Vec<StateModification>,
    },

    /// Resume with no changes
    Continue,
}

/// State modification request
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StateModification {
    /// Field path (e.g., "user.name", "settings.0.value")
    pub path: String,
    /// Operation type
    pub operation: ModificationOperation,
    /// New value for set/add operations
    pub value: Option<serde_json::Value>,
}

/// State modification operations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ModificationOperation {
    /// Set a field value
    Set,
    /// Delete a field
    Delete,
    /// Add to array
    Add,
    /// Remove from array
    Remove,
}

/// Interrupt point configuration
#[derive(Debug, Clone)]
pub struct InterruptConfig {
    /// Nodes to pause before
    pub pause_before: Vec<String>,
    /// Nodes to pause after
    pub pause_after: Vec<String>,
    /// Nodes requiring approval
    pub approval_nodes: HashMap<String, ApprovalConfig>,
    /// Timeout for waiting for response
    pub response_timeout: Option<std::time::Duration>,
}

impl Default for InterruptConfig {
    fn default() -> Self {
        Self {
            pause_before: Vec::new(),
            pause_after: Vec::new(),
            approval_nodes: HashMap::new(),
            response_timeout: None,
        }
    }
}

impl InterruptConfig {
    /// Create new interrupt config
    pub fn new() -> Self {
        Self::default()
    }

    /// Add pause before node
    pub fn pause_before(mut self, node_name: impl Into<String>) -> Self {
        self.pause_before.push(node_name.into());
        self
    }

    /// Add pause after node
    pub fn pause_after(mut self, node_name: impl Into<String>) -> Self {
        self.pause_after.push(node_name.into());
        self
    }

    /// Add approval requirement for node
    pub fn require_approval(mut self, node_name: impl Into<String>, config: ApprovalConfig) -> Self {
        self.approval_nodes.insert(node_name.into(), config);
        self
    }

    /// Set response timeout
    pub fn with_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.response_timeout = Some(timeout);
        self
    }

    /// Check if should pause before node
    pub fn should_pause_before(&self, node_name: &str) -> bool {
        self.pause_before.iter().any(|n| n == node_name)
    }

    /// Check if should pause after node
    pub fn should_pause_after(&self, node_name: &str) -> bool {
        self.pause_after.iter().any(|n| n == node_name)
    }

    /// Check if node requires approval
    pub fn requires_approval(&self, node_name: &str) -> Option<&ApprovalConfig> {
        self.approval_nodes.get(node_name)
    }
}

/// Approval gate configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalConfig {
    /// Approval prompt/question
    pub prompt: String,
    /// Available options for approval
    pub options: Vec<String>,
    /// Default option if timeout occurs
    pub default_option: Option<String>,
    /// Whether to allow state modification
    pub allow_state_modification: bool,
}

impl ApprovalConfig {
    /// Create new approval config
    pub fn new(prompt: impl Into<String>, options: Vec<String>) -> Self {
        Self {
            prompt: prompt.into(),
            options,
            default_option: None,
            allow_state_modification: false,
        }
    }

    /// Set default option
    pub fn with_default(mut self, option: impl Into<String>) -> Self {
        self.default_option = Some(option.into());
        self
    }

    /// Allow state modification
    pub fn allow_modifications(mut self) -> Self {
        self.allow_state_modification = true;
        self
    }
}

/// Interrupt event with context
#[derive(Debug, Clone)]
pub struct InterruptEvent<S: State> {
    /// Current node
    pub node_name: String,
    /// Current state
    pub state: S,
    /// Interrupt signal
    pub signal: InterruptSignal,
    /// Checkpoint ID for resumption
    pub checkpoint_id: Option<String>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

impl<S: State> InterruptEvent<S> {
    /// Create new interrupt event
    pub fn new(node_name: impl Into<String>, state: S, signal: InterruptSignal) -> Self {
        Self {
            node_name: node_name.into(),
            state,
            signal,
            checkpoint_id: None,
            timestamp: Utc::now(),
        }
    }

    /// Set checkpoint ID
    pub fn with_checkpoint(mut self, checkpoint_id: impl Into<String>) -> Self {
        self.checkpoint_id = Some(checkpoint_id.into());
        self
    }
}

/// Interrupt handler for managing pause/resume
pub struct InterruptHandler<S: State> {
    /// Channel for sending interrupt events
    interrupt_tx: Arc<RwLock<Option<mpsc::Sender<InterruptEvent<S>>>>>,
    /// Channel for receiving responses
    response_rx: Arc<RwLock<Option<mpsc::Receiver<InterruptResponse>>>>,
    /// Configuration
    config: Arc<RwLock<InterruptConfig>>,
}

impl<S: State> InterruptHandler<S> {
    /// Create new interrupt handler
    pub fn new(config: InterruptConfig) -> (Self, mpsc::Receiver<InterruptEvent<S>>, mpsc::Sender<InterruptResponse>) {
        let (interrupt_tx, interrupt_rx) = mpsc::channel(32);
        let (response_tx, response_rx) = mpsc::channel(32);

        let handler = Self {
            interrupt_tx: Arc::new(RwLock::new(Some(interrupt_tx))),
            response_rx: Arc::new(RwLock::new(Some(response_rx))),
            config: Arc::new(RwLock::new(config)),
        };

        (handler, interrupt_rx, response_tx)
    }

    /// Send interrupt event
    pub async fn send_interrupt(&self, event: InterruptEvent<S>) -> Result<()> {
        let tx_lock = self.interrupt_tx.read().await;
        if let Some(tx) = tx_lock.as_ref() {
            tx.send(event)
                .await
                .map_err(|e| Error::invalid_operation(format!("Failed to send interrupt: {}", e)))?;
            Ok(())
        } else {
            Err(Error::invalid_operation("Interrupt handler not initialized"))
        }
    }

    /// Wait for response
    pub async fn wait_for_response(&self) -> Result<InterruptResponse> {
        let timeout = self.config.read().await.response_timeout;

        let mut rx_lock = self.response_rx.write().await;
        if let Some(rx) = rx_lock.as_mut() {
            if let Some(timeout_duration) = timeout {
                match tokio::time::timeout(timeout_duration, rx.recv()).await {
                    Ok(Some(response)) => Ok(response),
                    Ok(None) => Err(Error::invalid_operation("Response channel closed")),
                    Err(_) => Err(Error::invalid_operation("Response timeout")),
                }
            } else {
                rx.recv()
                    .await
                    .ok_or_else(|| Error::invalid_operation("Response channel closed"))
            }
        } else {
            Err(Error::invalid_operation("Response receiver not initialized"))
        }
    }

    /// Check if should interrupt before node
    pub async fn should_interrupt_before(&self, node_name: &str) -> bool {
        let config = self.config.read().await;
        config.should_pause_before(node_name) || config.requires_approval(node_name).is_some()
    }

    /// Check if should interrupt after node
    pub async fn should_interrupt_after(&self, node_name: &str) -> bool {
        self.config.read().await.should_pause_after(node_name)
    }

    /// Get approval config for node
    pub async fn get_approval_config(&self, node_name: &str) -> Option<ApprovalConfig> {
        self.config.read().await.requires_approval(node_name).cloned()
    }

    /// Update configuration
    pub async fn update_config(&self, config: InterruptConfig) {
        *self.config.write().await = config;
    }
}

impl<S: State> Clone for InterruptHandler<S> {
    fn clone(&self) -> Self {
        Self {
            interrupt_tx: Arc::clone(&self.interrupt_tx),
            response_rx: Arc::clone(&self.response_rx),
            config: Arc::clone(&self.config),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct TestState {
        value: i32,
    }

    #[test]
    fn test_interrupt_signal() {
        let signal = InterruptSignal::PauseBefore {
            node_name: "test".to_string(),
            reason: "Testing".to_string(),
        };

        assert_eq!(
            signal,
            InterruptSignal::PauseBefore {
                node_name: "test".to_string(),
                reason: "Testing".to_string(),
            }
        );
    }

    #[test]
    fn test_interrupt_config() {
        let config = InterruptConfig::new()
            .pause_before("node1")
            .pause_after("node2");

        assert!(config.should_pause_before("node1"));
        assert!(!config.should_pause_before("node2"));
        assert!(config.should_pause_after("node2"));
        assert!(!config.should_pause_after("node1"));
    }

    #[test]
    fn test_approval_config() {
        let config = ApprovalConfig::new(
            "Approve this action?",
            vec!["Yes".to_string(), "No".to_string()],
        )
        .with_default("No")
        .allow_modifications();

        assert_eq!(config.prompt, "Approve this action?");
        assert_eq!(config.options.len(), 2);
        assert_eq!(config.default_option, Some("No".to_string()));
        assert!(config.allow_state_modification);
    }

    #[tokio::test]
    async fn test_interrupt_handler() {
        let config = InterruptConfig::new().pause_before("test_node");
        let (handler, mut interrupt_rx, _response_tx) = InterruptHandler::<TestState>::new(config);

        assert!(handler.should_interrupt_before("test_node").await);
        assert!(!handler.should_interrupt_before("other_node").await);

        let event = InterruptEvent::new(
            "test_node",
            TestState { value: 42 },
            InterruptSignal::PauseBefore {
                node_name: "test_node".to_string(),
                reason: "Test".to_string(),
            },
        );

        handler.send_interrupt(event.clone()).await.unwrap();

        let received = interrupt_rx.recv().await.unwrap();
        assert_eq!(received.node_name, "test_node");
        assert_eq!(received.state.value, 42);
    }

    #[test]
    fn test_state_modification() {
        let modification = StateModification {
            path: "user.name".to_string(),
            operation: ModificationOperation::Set,
            value: Some(serde_json::json!("John Doe")),
        };

        assert_eq!(modification.path, "user.name");
        assert_eq!(modification.operation, ModificationOperation::Set);
        assert!(modification.value.is_some());
    }
}
