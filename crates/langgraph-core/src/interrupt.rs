//! Human-in-the-loop interrupt functionality
//!
//! Provides enterprise-grade interrupt mechanisms for pausing graph execution
//! to await human input, approval, or decision-making.

use crate::{Result, State};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot, RwLock};

/// Interrupt reason/type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum InterruptReason {
    /// Waiting for human approval to continue
    ApprovalRequired,
    /// Waiting for human input/data
    InputRequired { prompt: String },
    /// Waiting for human decision between options
    DecisionRequired { options: Vec<String> },
    /// Custom interrupt reason
    Custom(String),
}

/// Interrupt request containing execution context
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "S: State")]
pub struct Interrupt<S: State> {
    /// Unique interrupt ID
    pub id: String,
    /// Thread/session ID
    pub thread_id: String,
    /// Node that triggered the interrupt
    pub node_name: String,
    /// Current state at interrupt point
    pub state: S,
    /// Reason for the interrupt
    pub reason: InterruptReason,
    /// Timestamp when interrupt was created
    #[serde(with = "chrono::serde::ts_seconds")]
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl<S: State> Interrupt<S> {
    /// Create a new interrupt
    pub fn new(
        thread_id: impl Into<String>,
        node_name: impl Into<String>,
        state: S,
        reason: InterruptReason,
    ) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            thread_id: thread_id.into(),
            node_name: node_name.into(),
            state,
            reason,
            created_at: chrono::Utc::now(),
            metadata: HashMap::new(),
        }
    }

    /// Add metadata to the interrupt
    pub fn with_metadata(
        mut self,
        key: impl Into<String>,
        value: serde_json::Value,
    ) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

/// Response to an interrupt containing human input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterruptResponse {
    /// ID of the interrupt being responded to
    pub interrupt_id: String,
    /// Human's response data
    pub response: serde_json::Value,
    /// Whether execution should continue (true) or abort (false)
    pub should_continue: bool,
    /// Optional state modifications
    pub state_updates: Option<HashMap<String, serde_json::Value>>,
}

impl InterruptResponse {
    /// Create a response to continue execution
    pub fn approve(interrupt_id: impl Into<String>) -> Self {
        Self {
            interrupt_id: interrupt_id.into(),
            response: serde_json::json!({"approved": true}),
            should_continue: true,
            state_updates: None,
        }
    }

    /// Create a response to reject/abort execution
    pub fn reject(interrupt_id: impl Into<String>) -> Self {
        Self {
            interrupt_id: interrupt_id.into(),
            response: serde_json::json!({"approved": false}),
            should_continue: false,
            state_updates: None,
        }
    }

    /// Create a response with custom data
    pub fn with_data(interrupt_id: impl Into<String>, data: serde_json::Value) -> Self {
        Self {
            interrupt_id: interrupt_id.into(),
            response: data,
            should_continue: true,
            state_updates: None,
        }
    }

    /// Add state updates to the response
    pub fn with_state_updates(mut self, updates: HashMap<String, serde_json::Value>) -> Self {
        self.state_updates = Some(updates);
        self
    }
}

/// Channel for communicating interrupt responses
type InterruptChannel = (
    oneshot::Sender<InterruptResponse>,
    oneshot::Receiver<InterruptResponse>,
);

/// Manages interrupts for a graph execution
pub struct InterruptManager<S: State> {
    /// Active interrupts indexed by ID
    interrupts: Arc<RwLock<HashMap<String, Interrupt<S>>>>,
    /// Response channels indexed by interrupt ID
    channels: Arc<RwLock<HashMap<String, oneshot::Sender<InterruptResponse>>>>,
    /// Broadcast channel for new interrupts
    interrupt_tx: mpsc::UnboundedSender<Interrupt<S>>,
    interrupt_rx: Arc<RwLock<mpsc::UnboundedReceiver<Interrupt<S>>>>,
}

impl<S: State> InterruptManager<S> {
    /// Create a new interrupt manager
    pub fn new() -> Self {
        let (interrupt_tx, interrupt_rx) = mpsc::unbounded_channel();
        Self {
            interrupts: Arc::new(RwLock::new(HashMap::new())),
            channels: Arc::new(RwLock::new(HashMap::new())),
            interrupt_tx,
            interrupt_rx: Arc::new(RwLock::new(interrupt_rx)),
        }
    }

    /// Register an interrupt and wait for response
    pub async fn register_interrupt(
        &self,
        interrupt: Interrupt<S>,
    ) -> Result<InterruptResponse> {
        let interrupt_id = interrupt.id.clone();

        // Create response channel
        let (tx, rx) = oneshot::channel();

        // Store interrupt and channel
        {
            let mut interrupts = self.interrupts.write().await;
            interrupts.insert(interrupt_id.clone(), interrupt.clone());
        }
        {
            let mut channels = self.channels.write().await;
            channels.insert(interrupt_id.clone(), tx);
        }

        // Broadcast the interrupt
        let _ = self.interrupt_tx.send(interrupt);

        // Wait for response
        let response = rx.await.map_err(|e| {
            crate::Error::invalid_operation(format!("Interrupt response channel closed: {}", e))
        })?;

        // Clean up
        {
            let mut interrupts = self.interrupts.write().await;
            interrupts.remove(&interrupt_id);
        }

        Ok(response)
    }

    /// Respond to an interrupt
    pub async fn respond(&self, response: InterruptResponse) -> Result<()> {
        let mut channels = self.channels.write().await;

        if let Some(tx) = channels.remove(&response.interrupt_id) {
            tx.send(response).map_err(|_| {
                crate::Error::invalid_operation("Failed to send interrupt response")
            })?;
            Ok(())
        } else {
            Err(crate::Error::invalid_operation(format!(
                "No active interrupt with ID: {}",
                response.interrupt_id
            )))
        }
    }

    /// Get all active interrupts
    pub async fn get_active_interrupts(&self) -> Vec<Interrupt<S>> {
        let interrupts = self.interrupts.read().await;
        interrupts.values().cloned().collect()
    }

    /// Get a specific interrupt by ID
    pub async fn get_interrupt(&self, interrupt_id: &str) -> Option<Interrupt<S>> {
        let interrupts = self.interrupts.read().await;
        interrupts.get(interrupt_id).cloned()
    }

    /// Subscribe to new interrupts
    pub async fn subscribe(&self) -> mpsc::UnboundedReceiver<Interrupt<S>> {
        let (tx, rx) = mpsc::unbounded_channel();

        // Forward interrupts to the new subscriber
        let interrupt_rx = self.interrupt_rx.clone();
        tokio::spawn(async move {
            let mut interrupt_rx = interrupt_rx.write().await;
            while let Some(interrupt) = interrupt_rx.recv().await {
                if tx.send(interrupt).is_err() {
                    break;
                }
            }
        });

        rx
    }

    /// Clear all active interrupts
    pub async fn clear(&self) {
        let mut interrupts = self.interrupts.write().await;
        let mut channels = self.channels.write().await;
        interrupts.clear();
        channels.clear();
    }
}

impl<S: State> Default for InterruptManager<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: State> Clone for InterruptManager<S> {
    fn clone(&self) -> Self {
        Self {
            interrupts: self.interrupts.clone(),
            channels: self.channels.clone(),
            interrupt_tx: self.interrupt_tx.clone(),
            interrupt_rx: self.interrupt_rx.clone(),
        }
    }
}

/// Configuration for node-level interrupt behavior
#[derive(Debug, Clone)]
pub struct InterruptConfig {
    /// Whether this node should interrupt before execution
    pub interrupt_before: bool,
    /// Whether this node should interrupt after execution
    pub interrupt_after: bool,
    /// The reason for the interrupt
    pub reason: InterruptReason,
}

impl InterruptConfig {
    /// Create interrupt config that pauses before execution
    pub fn before(reason: InterruptReason) -> Self {
        Self {
            interrupt_before: true,
            interrupt_after: false,
            reason,
        }
    }

    /// Create interrupt config that pauses after execution
    pub fn after(reason: InterruptReason) -> Self {
        Self {
            interrupt_before: false,
            interrupt_after: true,
            reason,
        }
    }

    /// Create interrupt config that pauses both before and after
    pub fn both(reason: InterruptReason) -> Self {
        Self {
            interrupt_before: true,
            interrupt_after: true,
            reason,
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
    async fn test_interrupt_creation() {
        let state = TestState { value: 42 };
        let interrupt = Interrupt::new(
            "thread-1",
            "test-node",
            state.clone(),
            InterruptReason::ApprovalRequired,
        );

        assert_eq!(interrupt.thread_id, "thread-1");
        assert_eq!(interrupt.node_name, "test-node");
        assert_eq!(interrupt.state.value, 42);
        assert_eq!(interrupt.reason, InterruptReason::ApprovalRequired);
    }

    #[tokio::test]
    async fn test_interrupt_manager() {
        let manager = InterruptManager::<TestState>::new();
        let state = TestState { value: 42 };

        let interrupt = Interrupt::new(
            "thread-1",
            "test-node",
            state.clone(),
            InterruptReason::ApprovalRequired,
        );
        let interrupt_id = interrupt.id.clone();

        // Register interrupt in background
        let manager_clone = manager.clone();
        let register_handle = tokio::spawn(async move {
            manager_clone.register_interrupt(interrupt).await
        });

        // Give it a moment to register
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        // Check active interrupts
        let active = manager.get_active_interrupts().await;
        assert_eq!(active.len(), 1);

        // Respond to interrupt
        let response = InterruptResponse::approve(&interrupt_id);
        manager.respond(response).await.unwrap();

        // Wait for registration to complete
        let result = register_handle.await.unwrap().unwrap();
        assert!(result.should_continue);
    }

    #[test]
    fn test_interrupt_response() {
        let response = InterruptResponse::approve("test-id");
        assert!(response.should_continue);

        let response = InterruptResponse::reject("test-id");
        assert!(!response.should_continue);

        let data = serde_json::json!({"key": "value"});
        let response = InterruptResponse::with_data("test-id", data);
        assert!(response.should_continue);
        assert_eq!(response.response["key"], "value");
    }
}
