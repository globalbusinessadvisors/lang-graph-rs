//! Human-in-the-loop interactive execution
//!
//! This module provides interactive graph execution with pause/resume capabilities,
//! allowing human intervention at strategic points in the graph execution flow.

use crate::{Error, Result, State};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};

/// Strategy for when to interrupt execution
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum InterruptStrategy {
    /// Interrupt before executing specific nodes
    BeforeNodes(HashSet<String>),
    /// Interrupt after executing specific nodes
    AfterNodes(HashSet<String>),
    /// Interrupt before and after specific nodes
    BeforeAndAfterNodes(HashSet<String>),
    /// Interrupt on any error
    OnError,
    /// Interrupt every N steps
    EveryNSteps(usize),
    /// Custom interrupt based on state condition
    Custom,
}

impl Default for InterruptStrategy {
    fn default() -> Self {
        InterruptStrategy::BeforeNodes(HashSet::new())
    }
}

/// Point at which execution was interrupted
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionPoint {
    /// Name of the node where execution paused
    pub node_name: String,
    /// Whether the pause was before or after node execution
    pub position: InteractionPosition,
    /// Current step count
    pub step_count: usize,
    /// Timestamp of the interruption
    pub timestamp: std::time::SystemTime,
    /// Reason for the interruption
    pub reason: InterruptReason,
}

/// Position relative to node execution
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InteractionPosition {
    /// Before node execution
    Before,
    /// After node execution
    After,
}

/// Reason for interruption
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InterruptReason {
    /// Manual breakpoint at node
    Breakpoint,
    /// Error occurred during execution
    Error(String),
    /// Step interval reached
    StepInterval,
    /// Custom condition met
    Custom(String),
}

/// Action to take after human intervention
#[derive(Debug, Clone)]
pub enum InteractionResponse<S: State> {
    /// Continue with the current state
    Continue,
    /// Continue with a modified state
    ContinueWith(S),
    /// Skip the current node and continue
    Skip,
    /// Abort execution
    Abort(String),
    /// Resume with a different node
    ResumeAt(String),
}

/// Configuration for interactive execution
#[derive(Debug, Clone)]
pub struct InteractiveConfig {
    /// Interrupt strategies to apply
    pub strategies: Vec<InterruptStrategy>,
    /// Maximum number of steps before forcing termination
    pub max_steps: Option<usize>,
    /// Enable automatic checkpointing at interruption points
    pub auto_checkpoint: bool,
    /// Thread ID for checkpoint grouping
    pub thread_id: Option<String>,
    /// Enable debug logging
    pub debug: bool,
}

impl Default for InteractiveConfig {
    fn default() -> Self {
        Self {
            strategies: vec![],
            max_steps: Some(1000),
            auto_checkpoint: false,
            thread_id: None,
            debug: false,
        }
    }
}

impl InteractiveConfig {
    /// Create config with breakpoints before specific nodes
    pub fn with_breakpoints_before(nodes: impl IntoIterator<Item = impl Into<String>>) -> Self {
        let node_set: HashSet<String> = nodes.into_iter().map(|n| n.into()).collect();
        Self {
            strategies: vec![InterruptStrategy::BeforeNodes(node_set)],
            ..Default::default()
        }
    }

    /// Create config with breakpoints after specific nodes
    pub fn with_breakpoints_after(nodes: impl IntoIterator<Item = impl Into<String>>) -> Self {
        let node_set: HashSet<String> = nodes.into_iter().map(|n| n.into()).collect();
        Self {
            strategies: vec![InterruptStrategy::AfterNodes(node_set)],
            ..Default::default()
        }
    }

    /// Enable automatic checkpointing
    pub fn with_checkpointing(mut self, thread_id: impl Into<String>) -> Self {
        self.auto_checkpoint = true;
        self.thread_id = Some(thread_id.into());
        self
    }

    /// Set maximum steps
    pub fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = Some(max_steps);
        self
    }

    /// Add an interrupt strategy
    pub fn with_strategy(mut self, strategy: InterruptStrategy) -> Self {
        self.strategies.push(strategy);
        self
    }
}

/// Execution event for tracking history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionEvent {
    /// Name of the node that was executed
    pub node_name: String,
    /// Step number in the execution
    pub step: usize,
    /// Timestamp when the node was executed
    pub timestamp: std::time::SystemTime,
    /// Whether the node execution was successful
    pub success: bool,
    /// Error message if execution failed
    pub error: Option<String>,
}

/// Execution trace for tracking graph traversal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionTrace {
    /// Ordered list of execution events
    pub events: Vec<ExecutionEvent>,
    /// Total steps executed
    pub total_steps: usize,
    /// Start time of execution
    pub start_time: std::time::SystemTime,
    /// End time of execution (if completed)
    pub end_time: Option<std::time::SystemTime>,
}

impl ExecutionTrace {
    /// Create a new execution trace
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            total_steps: 0,
            start_time: std::time::SystemTime::now(),
            end_time: None,
        }
    }

    /// Add an event to the trace
    pub fn add_event(&mut self, event: ExecutionEvent) {
        self.events.push(event);
        self.total_steps += 1;
    }

    /// Mark execution as completed
    pub fn complete(&mut self) {
        self.end_time = Some(std::time::SystemTime::now());
    }

    /// Get nodes in execution order
    pub fn node_sequence(&self) -> Vec<String> {
        self.events.iter().map(|e| e.node_name.clone()).collect()
    }

    /// Get failed nodes
    pub fn failed_nodes(&self) -> Vec<String> {
        self.events
            .iter()
            .filter(|e| !e.success)
            .map(|e| e.node_name.clone())
            .collect()
    }
}

impl Default for ExecutionTrace {
    fn default() -> Self {
        Self::new()
    }
}

/// Handle for controlling interactive execution
pub struct ExecutionHandle<S: State> {
    /// Channel for receiving interaction points
    interaction_rx: mpsc::UnboundedReceiver<(InteractionPoint, S)>,
    /// Channel for sending responses
    response_tx: mpsc::UnboundedSender<InteractionResponse<S>>,
    /// Execution trace
    trace: Arc<RwLock<ExecutionTrace>>,
    /// Current interaction point
    current_interaction: Option<(InteractionPoint, S)>,
}

impl<S: State> ExecutionHandle<S> {
    /// Create a new execution handle
    pub(crate) fn new(
        interaction_rx: mpsc::UnboundedReceiver<(InteractionPoint, S)>,
        response_tx: mpsc::UnboundedSender<InteractionResponse<S>>,
        trace: Arc<RwLock<ExecutionTrace>>,
    ) -> Self {
        Self {
            interaction_rx,
            response_tx,
            trace,
            current_interaction: None,
        }
    }

    /// Wait for the next interaction point (blocking)
    pub async fn wait_for_interaction(&mut self) -> Option<(InteractionPoint, S)> {
        self.current_interaction = self.interaction_rx.recv().await;
        self.current_interaction.clone()
    }

    /// Get the current interaction point without blocking
    pub fn current(&self) -> Option<&(InteractionPoint, S)> {
        self.current_interaction.as_ref()
    }

    /// Continue execution with the current state
    pub fn continue_execution(&self) -> Result<()> {
        self.response_tx
            .send(InteractionResponse::Continue)
            .map_err(|_| Error::invalid_operation("Failed to send response"))
    }

    /// Continue execution with a modified state
    pub fn continue_with(&self, state: S) -> Result<()> {
        self.response_tx
            .send(InteractionResponse::ContinueWith(state))
            .map_err(|_| Error::invalid_operation("Failed to send response"))
    }

    /// Skip the current node and continue
    pub fn skip(&self) -> Result<()> {
        self.response_tx
            .send(InteractionResponse::Skip)
            .map_err(|_| Error::invalid_operation("Failed to send response"))
    }

    /// Abort execution with a reason
    pub fn abort(&self, reason: impl Into<String>) -> Result<()> {
        self.response_tx
            .send(InteractionResponse::Abort(reason.into()))
            .map_err(|_| Error::invalid_operation("Failed to send response"))
    }

    /// Resume execution at a different node
    pub fn resume_at(&self, node: impl Into<String>) -> Result<()> {
        self.response_tx
            .send(InteractionResponse::ResumeAt(node.into()))
            .map_err(|_| Error::invalid_operation("Failed to send response"))
    }

    /// Get the execution trace
    pub async fn trace(&self) -> ExecutionTrace {
        self.trace.read().await.clone()
    }

    /// Get a summary of the execution trace
    pub async fn trace_summary(&self) -> String {
        let trace = self.trace.read().await;
        format!(
            "Steps: {}, Nodes: {:?}",
            trace.total_steps,
            trace.node_sequence()
        )
    }
}

/// Internal state for interactive execution
pub(crate) struct InteractiveState<S: State> {
    pub(crate) interaction_tx: mpsc::UnboundedSender<(InteractionPoint, S)>,
    pub(crate) response_rx: mpsc::UnboundedReceiver<InteractionResponse<S>>,
    pub(crate) trace: Arc<RwLock<ExecutionTrace>>,
    pub(crate) config: InteractiveConfig,
    pub(crate) step_count: usize,
}

impl<S: State> InteractiveState<S> {
    /// Check if execution should be interrupted at this point
    pub(crate) fn should_interrupt(
        &self,
        node_name: &str,
        position: InteractionPosition,
    ) -> Option<InterruptReason> {
        for strategy in &self.config.strategies {
            match strategy {
                InterruptStrategy::BeforeNodes(nodes) => {
                    if position == InteractionPosition::Before && nodes.contains(node_name) {
                        return Some(InterruptReason::Breakpoint);
                    }
                }
                InterruptStrategy::AfterNodes(nodes) => {
                    if position == InteractionPosition::After && nodes.contains(node_name) {
                        return Some(InterruptReason::Breakpoint);
                    }
                }
                InterruptStrategy::BeforeAndAfterNodes(nodes) => {
                    if nodes.contains(node_name) {
                        return Some(InterruptReason::Breakpoint);
                    }
                }
                InterruptStrategy::EveryNSteps(n) => {
                    if self.step_count > 0 && self.step_count % n == 0 {
                        return Some(InterruptReason::StepInterval);
                    }
                }
                InterruptStrategy::OnError => {
                    // Handled separately when errors occur
                }
                InterruptStrategy::Custom => {
                    // Custom interrupts handled by caller
                }
            }
        }
        None
    }

    /// Send an interaction point and wait for response
    pub(crate) async fn interact(
        &mut self,
        node_name: String,
        position: InteractionPosition,
        state: S,
        reason: InterruptReason,
    ) -> Result<InteractionResponse<S>> {
        let interaction = InteractionPoint {
            node_name,
            position,
            step_count: self.step_count,
            timestamp: std::time::SystemTime::now(),
            reason,
        };

        self.interaction_tx
            .send((interaction, state))
            .map_err(|_| Error::invalid_operation("Interaction channel closed"))?;

        self.response_rx
            .recv()
            .await
            .ok_or_else(|| Error::invalid_operation("Response channel closed"))
    }

    /// Record an execution event
    pub(crate) async fn record_event(&self, event: ExecutionEvent) {
        let mut trace = self.trace.write().await;
        trace.add_event(event);
    }
}

/// Create channels for interactive execution
pub(crate) fn create_channels<S: State>() -> (
    ExecutionHandle<S>,
    InteractiveState<S>,
    InteractiveConfig,
) {
    let (interaction_tx, interaction_rx) = mpsc::unbounded_channel();
    let (response_tx, response_rx) = mpsc::unbounded_channel();
    let trace = Arc::new(RwLock::new(ExecutionTrace::new()));
    let config = InteractiveConfig::default();

    let handle = ExecutionHandle::new(interaction_rx, response_tx, trace.clone());
    let state = InteractiveState {
        interaction_tx,
        response_rx,
        trace,
        config: config.clone(),
        step_count: 0,
    };

    (handle, state, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    struct TestState {
        value: i32,
    }

    #[test]
    fn test_interactive_config() {
        let config = InteractiveConfig::with_breakpoints_before(vec!["node1", "node2"])
            .with_max_steps(100)
            .with_checkpointing("test-thread");

        assert_eq!(config.strategies.len(), 1);
        assert_eq!(config.max_steps, Some(100));
        assert!(config.auto_checkpoint);
        assert_eq!(config.thread_id, Some("test-thread".to_string()));
    }

    #[test]
    fn test_execution_trace() {
        let mut trace = ExecutionTrace::new();
        trace.add_event(ExecutionEvent {
            node_name: "node1".to_string(),
            step: 1,
            timestamp: std::time::SystemTime::now(),
            success: true,
            error: None,
        });
        trace.add_event(ExecutionEvent {
            node_name: "node2".to_string(),
            step: 2,
            timestamp: std::time::SystemTime::now(),
            success: false,
            error: Some("Test error".to_string()),
        });

        assert_eq!(trace.total_steps, 2);
        assert_eq!(trace.node_sequence(), vec!["node1", "node2"]);
        assert_eq!(trace.failed_nodes(), vec!["node2"]);
    }
}
