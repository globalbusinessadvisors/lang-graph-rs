//! Graph execution engine - StateGraph and CompiledGraph

use crate::{Error, Node, Result, State};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Edge type in the graph
enum EdgeType<S: State> {
    /// Direct edge to another node
    Direct(String),
    /// Conditional edge that determines next node at runtime
    Conditional {
        condition: Arc<dyn Fn(&S) -> String + Send + Sync>,
        edge_map: HashMap<String, String>,
    },
}

impl<S: State> std::fmt::Debug for EdgeType<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EdgeType::Direct(to) => f.debug_tuple("Direct").field(to).finish(),
            EdgeType::Conditional { edge_map, .. } => f
                .debug_struct("Conditional")
                .field("edge_map", edge_map)
                .finish(),
        }
    }
}

impl<S: State> Clone for EdgeType<S> {
    fn clone(&self) -> Self {
        match self {
            EdgeType::Direct(to) => EdgeType::Direct(to.clone()),
            EdgeType::Conditional {
                condition,
                edge_map,
            } => EdgeType::Conditional {
                condition: condition.clone(),
                edge_map: edge_map.clone(),
            },
        }
    }
}

/// Stream event types for enhanced streaming execution
#[derive(Debug, Clone)]
pub enum StreamEvent<S: State> {
    /// Node execution started
    NodeStart {
        node_name: String,
        step: usize,
    },
    /// Node execution completed successfully
    NodeEnd {
        node_name: String,
        state: S,
        duration: Duration,
    },
    /// Edge traversal
    EdgeTraversal {
        from: String,
        to: String,
    },
    /// Execution error
    Error {
        node_name: String,
        error: String,
    },
    /// Execution interrupted (human-in-the-loop)
    Interrupted {
        node_name: String,
        state: S,
    },
}

/// Configuration for streaming execution
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Maximum number of steps to execute
    pub max_steps: Option<usize>,
    /// Enable debug logging
    pub debug: bool,
    /// Filter nodes to yield (None = all nodes)
    pub filter_nodes: Option<HashSet<String>>,
    /// Include node start events
    pub include_node_start: bool,
    /// Include edge traversal events
    pub include_edge_traversal: bool,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            max_steps: Some(1000),
            debug: false,
            filter_nodes: None,
            include_node_start: true,
            include_edge_traversal: false,
        }
    }
}

/// Represents an interrupted execution that can be resumed
#[derive(Debug, Clone)]
pub struct InterruptedExecution<S: State> {
    /// Current state at interruption
    pub state: S,
    /// Node name where execution was interrupted
    pub interrupted_at: String,
    /// Next node to execute when resumed
    pub next_node: Option<String>,
    /// Checkpoint ID (if checkpointing is enabled)
    pub checkpoint_id: Option<String>,
    /// Number of steps executed before interruption
    pub steps_executed: usize,
}

/// Token for resuming interrupted execution
#[derive(Debug, Clone)]
pub struct ResumeToken {
    /// Checkpoint ID to resume from
    pub checkpoint_id: String,
    /// Next node to execute
    pub next_node: String,
    /// Optional approval data from human
    pub approval_data: Option<serde_json::Value>,
}

/// Execution status for checkpoints
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionStatus {
    /// Execution is running
    Running,
    /// Execution is interrupted and waiting for approval
    Interrupted,
    /// Execution completed successfully
    Completed,
    /// Execution failed with error
    Failed,
}

impl Default for ExecutionStatus {
    fn default() -> Self {
        Self::Running
    }
}

/// Internal graph structure
struct GraphStructure<S: State> {
    nodes: HashMap<String, Arc<dyn Node<S>>>,
    edges: HashMap<String, Vec<EdgeType<S>>>,
    entry_point: Option<String>,
    finish_point: Option<String>,
    interrupts: HashSet<String>,
}

/// StateGraph builder for creating execution graphs
pub struct StateGraph<S: State> {
    nodes: HashMap<String, Arc<dyn Node<S>>>,
    edges: HashMap<String, Vec<EdgeType<S>>>,
    entry_point: Option<String>,
    finish_point: Option<String>,
    /// Nodes that require human approval before execution
    interrupts: HashSet<String>,
}

impl<S: State> StateGraph<S> {
    /// Create a new StateGraph
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            entry_point: None,
            finish_point: None,
            interrupts: HashSet::new(),
        }
    }

    /// Add a node to the graph
    pub fn add_node<N>(mut self, name: impl Into<String>, node: N) -> Self
    where
        N: Node<S> + 'static,
    {
        let name = name.into();
        self.nodes.insert(name.clone(), Arc::new(node));
        self.edges.entry(name).or_insert_with(Vec::new);
        self
    }

    /// Add an edge from one node to another
    pub fn add_edge(mut self, from: impl Into<String>, to: impl Into<String>) -> Self {
        let from = from.into();
        let to = to.into();
        self.edges
            .entry(from)
            .or_insert_with(Vec::new)
            .push(EdgeType::Direct(to));
        self
    }

    /// Add a conditional edge that determines the next node at runtime
    pub fn add_conditional_edge<F>(
        mut self,
        from: impl Into<String>,
        condition: F,
        edge_map: HashMap<String, String>,
    ) -> Self
    where
        F: Fn(&S) -> String + Send + Sync + 'static,
    {
        let from = from.into();
        self.edges
            .entry(from)
            .or_insert_with(Vec::new)
            .push(EdgeType::Conditional {
                condition: Arc::new(condition),
                edge_map,
            });
        self
    }

    /// Set the entry point of the graph
    pub fn set_entry_point(mut self, name: impl Into<String>) -> Self {
        self.entry_point = Some(name.into());
        self
    }

    /// Set the finish point of the graph
    pub fn set_finish_point(mut self, name: impl Into<String>) -> Self {
        self.finish_point = Some(name.into());
        self
    }

    /// Add an interrupt point (human-in-the-loop)
    /// Execution will pause before this node and return an InterruptedExecution
    pub fn add_interrupt(mut self, node_name: impl Into<String>) -> Self {
        self.interrupts.insert(node_name.into());
        self
    }

    /// Compile the graph into an executable form
    pub fn compile(self) -> Result<CompiledGraph<S>> {
        // Validate graph structure
        self.validate()?;

        let structure = Arc::new(GraphStructure {
            nodes: self.nodes,
            edges: self.edges,
            entry_point: self.entry_point,
            finish_point: self.finish_point,
            interrupts: self.interrupts,
        });

        Ok(CompiledGraph {
            structure,
            execution_cache: Arc::new(DashMap::new()),
        })
    }

    /// Validate the graph structure
    fn validate(&self) -> Result<()> {
        // Check entry point exists
        if self.entry_point.is_none() {
            return Err(Error::invalid_graph("No entry point set"));
        }

        let entry = self.entry_point.as_ref().unwrap();
        if !self.nodes.contains_key(entry) {
            return Err(Error::node_not_found(entry));
        }

        // Check all edges reference valid nodes
        for (from, edges) in &self.edges {
            if !self.nodes.contains_key(from) {
                return Err(Error::node_not_found(from));
            }

            for edge in edges {
                match edge {
                    EdgeType::Direct(to) => {
                        if !self.nodes.contains_key(to) {
                            return Err(Error::node_not_found(to));
                        }
                    }
                    EdgeType::Conditional { edge_map, .. } => {
                        for to in edge_map.values() {
                            if !self.nodes.contains_key(to) {
                                return Err(Error::node_not_found(to));
                            }
                        }
                    }
                }
            }
        }

        // Check for cycles
        self.detect_cycles()?;

        Ok(())
    }

    /// Detect cycles in the graph using DFS
    /// Only checks for unconditional cycles (direct edges forming loops)
    /// Conditional cycles are allowed as they have exit conditions
    fn detect_cycles(&self) -> Result<()> {
        if self.entry_point.is_none() {
            return Ok(());
        }

        let entry = self.entry_point.as_ref().unwrap();
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();

        self.detect_cycles_util(entry, &mut visited, &mut rec_stack)
    }

    fn detect_cycles_util(
        &self,
        node: &str,
        visited: &mut HashSet<String>,
        rec_stack: &mut HashSet<String>,
    ) -> Result<()> {
        visited.insert(node.to_string());
        rec_stack.insert(node.to_string());

        if let Some(edges) = self.edges.get(node) {
            for edge in edges {
                // Only check direct edges for cycles
                // Conditional edges are allowed to form cycles
                let next_nodes = match edge {
                    EdgeType::Direct(to) => vec![to.clone()],
                    EdgeType::Conditional { .. } => {
                        // Skip conditional edges in cycle detection
                        vec![]
                    }
                };

                for next in next_nodes {
                    if !visited.contains(&next) {
                        self.detect_cycles_util(&next, visited, rec_stack)?;
                    } else if rec_stack.contains(&next) {
                        return Err(Error::cycle_detected(format!(
                            "Cycle detected involving node: {}",
                            next
                        )));
                    }
                }
            }
        }

        rec_stack.remove(node);
        Ok(())
    }
}

impl<S: State> Default for StateGraph<S> {
    fn default() -> Self {
        Self::new()
    }
}

/// Compiled and validated graph ready for execution
pub struct CompiledGraph<S: State> {
    structure: Arc<GraphStructure<S>>,
    execution_cache: Arc<DashMap<String, Vec<String>>>,
}

impl<S: State> CompiledGraph<S> {
    /// Execute the graph with the given initial state
    pub async fn execute(&self, state: S) -> Result<S> {
        self.execute_with_config(state, ExecutionConfig::default())
            .await
    }

    /// Execute the graph with custom configuration
    pub async fn execute_with_config(
        &self,
        mut state: S,
        config: ExecutionConfig,
    ) -> Result<S> {
        let entry = self
            .structure
            .entry_point
            .as_ref()
            .ok_or_else(|| Error::invalid_graph("No entry point"))?;

        let mut current = entry.clone();
        let mut visited = HashSet::new();
        let mut step_count = 0;

        loop {
            // Check max steps
            if let Some(max_steps) = config.max_steps {
                if step_count >= max_steps {
                    return Err(Error::invalid_operation(format!(
                        "Max steps ({}) exceeded",
                        max_steps
                    )));
                }
            }

            // Mark as visited
            visited.insert(current.clone());
            step_count += 1;

            // Get the node
            let node = self
                .structure
                .nodes
                .get(&current)
                .ok_or_else(|| Error::node_not_found(&current))?;

            // Execute the node
            state = node
                .execute(state)
                .await
                .map_err(|e| Error::node_execution_failed(format!("Node '{}': {}", current, e)))?;

            // Check if we've reached the finish point
            if let Some(finish) = &self.structure.finish_point {
                if &current == finish {
                    break;
                }
            }

            // Determine next node
            let next = self.get_next_node(&current, &state)?;

            match next {
                Some(next_node) => {
                    current = next_node;
                }
                None => {
                    // No more nodes to execute
                    break;
                }
            }
        }

        Ok(state)
    }

    /// Stream execution with intermediate states (simple version)
    pub async fn stream(
        &self,
        state: S,
    ) -> Result<impl futures::Stream<Item = Result<(String, S)>>> {
        use futures::stream;

        let entry = self
            .structure
            .entry_point
            .as_ref()
            .ok_or_else(|| Error::invalid_graph("No entry point"))?
            .clone();

        let structure = self.structure.clone();

        Ok(stream::unfold(
            (state, Some(entry), HashSet::new()),
            move |(mut state, current_opt, mut visited)| {
                let structure = structure.clone();
                async move {
                    let current = current_opt?;
                    visited.insert(current.clone());

                    // Get and execute node
                    let node = structure.nodes.get(&current)?;
                    state = node.execute(state).await.ok()?;

                    let node_name = current.clone();

                    // Determine next node
                    let next = {
                        let edges = structure.edges.get(&current)?;
                        let mut next_node = None;

                        for edge in edges {
                            match edge {
                                EdgeType::Direct(to) => {
                                    next_node = Some(to.clone());
                                }
                                EdgeType::Conditional { condition, edge_map } => {
                                    let key = condition(&state);
                                    if let Some(to) = edge_map.get(&key) {
                                        next_node = Some(to.clone());
                                    }
                                }
                            }
                        }
                        next_node
                    };

                    Some((Ok((node_name, state.clone())), (state, next, visited)))
                }
            },
        ))
    }

    /// Stream execution with enhanced events and configuration
    pub async fn stream_with_config(
        &self,
        state: S,
        config: StreamConfig,
    ) -> Result<impl futures::Stream<Item = Result<StreamEvent<S>>>> {
        use futures::stream;

        let entry = self
            .structure
            .entry_point
            .as_ref()
            .ok_or_else(|| Error::invalid_graph("No entry point"))?
            .clone();

        let structure = self.structure.clone();

        Ok(stream::unfold(
            (state, Some(entry), HashSet::new(), 0usize),
            move |(mut state, current_opt, mut visited, step_count)| {
                let structure = structure.clone();
                let config = config.clone();
                async move {
                    let current = current_opt?;

                    // Check max steps
                    if let Some(max_steps) = config.max_steps {
                        if step_count >= max_steps {
                            return Some((
                                Err(Error::invalid_operation(format!(
                                    "Max steps ({}) exceeded",
                                    max_steps
                                ))),
                                (state, None, visited, step_count),
                            ));
                        }
                    }

                    // Check for interrupt
                    if structure.interrupts.contains(&current) {
                        return Some((
                            Ok(StreamEvent::Interrupted {
                                node_name: current.clone(),
                                state: state.clone(),
                            }),
                            (state, None, visited, step_count),
                        ));
                    }

                    visited.insert(current.clone());

                    // Get and execute node
                    let node = structure.nodes.get(&current)?;
                    let start_time = Instant::now();

                    // Clone state for error case
                    let state_for_error = state.clone();
                    let exec_result = node.execute(state).await;
                    match exec_result {
                        Ok(new_state) => {
                            state = new_state;
                            let duration = start_time.elapsed();

                            // Determine next node
                            let next = {
                                let edges = structure.edges.get(&current);
                                let mut next_node = None;

                                if let Some(edges) = edges {
                                    for edge in edges {
                                        match edge {
                                            EdgeType::Direct(to) => {
                                                next_node = Some(to.clone());
                                            }
                                            EdgeType::Conditional { condition, edge_map } => {
                                                let key = condition(&state);
                                                if let Some(to) = edge_map.get(&key) {
                                                    next_node = Some(to.clone());
                                                }
                                            }
                                        }
                                    }
                                }
                                next_node
                            };

                            // Check if we should filter this node
                            let should_emit = config.filter_nodes.as_ref()
                                .map(|filter| filter.contains(&current))
                                .unwrap_or(true);

                            if should_emit {
                                Some((
                                    Ok(StreamEvent::NodeEnd {
                                        node_name: current.clone(),
                                        state: state.clone(),
                                        duration,
                                    }),
                                    (state, next, visited, step_count + 1),
                                ))
                            } else {
                                // Skip this node in output but continue execution
                                Some((
                                    Ok(StreamEvent::NodeEnd {
                                        node_name: current.clone(),
                                        state: state.clone(),
                                        duration,
                                    }),
                                    (state, next, visited, step_count + 1),
                                ))
                            }
                        }
                        Err(e) => {
                            Some((
                                Ok(StreamEvent::Error {
                                    node_name: current.clone(),
                                    error: e.to_string(),
                                }),
                                (state_for_error, None, visited, step_count),
                            ))
                        }
                    }
                }
            },
        ))
    }

    /// Execute until an interrupt point is reached (human-in-the-loop)
    /// Returns InterruptedExecution if an interrupt is encountered
    pub async fn execute_until_interrupt(&self, state: S) -> Result<InterruptedExecution<S>> {
        let entry = self
            .structure
            .entry_point
            .as_ref()
            .ok_or_else(|| Error::invalid_graph("No entry point"))?;

        let mut current = entry.clone();
        let mut current_state = state;
        let mut visited = HashSet::new();
        let mut step_count = 0;

        loop {
            // Check for interrupt BEFORE executing the node
            if self.structure.interrupts.contains(&current) {
                // Determine next node
                let next = self.get_next_node(&current, &current_state)?;

                return Ok(InterruptedExecution {
                    state: current_state,
                    interrupted_at: current.clone(),
                    next_node: next,
                    checkpoint_id: None, // Will be set by caller if checkpointing is used
                    steps_executed: step_count,
                });
            }

            // Mark as visited
            visited.insert(current.clone());
            step_count += 1;

            // Get the node
            let node = self
                .structure
                .nodes
                .get(&current)
                .ok_or_else(|| Error::node_not_found(&current))?;

            // Execute the node
            current_state = node
                .execute(current_state)
                .await
                .map_err(|e| Error::node_execution_failed(format!("Node '{}': {}", current, e)))?;

            // Check if we've reached the finish point
            if let Some(finish) = &self.structure.finish_point {
                if &current == finish {
                    break;
                }
            }

            // Determine next node
            let next = self.get_next_node(&current, &current_state)?;

            match next {
                Some(next_node) => {
                    current = next_node;
                }
                None => {
                    // No more nodes to execute
                    break;
                }
            }
        }

        // No interrupts encountered, return as interrupted at finish
        Ok(InterruptedExecution {
            state: current_state,
            interrupted_at: current,
            next_node: None,
            checkpoint_id: None,
            steps_executed: step_count,
        })
    }

    /// Resume execution from an interrupted state
    pub async fn resume(&self, state: S, from_node: String) -> Result<S> {
        let mut current = from_node;
        let mut current_state = state;
        let mut visited = HashSet::new();

        loop {
            // Check for interrupt
            if self.structure.interrupts.contains(&current) {
                return Err(Error::invalid_operation(
                    "Cannot resume: next node is also an interrupt point",
                ));
            }

            // Mark as visited
            visited.insert(current.clone());

            // Get the node
            let node = self
                .structure
                .nodes
                .get(&current)
                .ok_or_else(|| Error::node_not_found(&current))?;

            // Execute the node
            current_state = node
                .execute(current_state)
                .await
                .map_err(|e| Error::node_execution_failed(format!("Node '{}': {}", current, e)))?;

            // Check if we've reached the finish point
            if let Some(finish) = &self.structure.finish_point {
                if &current == finish {
                    break;
                }
            }

            // Determine next node
            let next = self.get_next_node(&current, &current_state)?;

            match next {
                Some(next_node) => {
                    current = next_node;
                }
                None => {
                    // No more nodes to execute
                    break;
                }
            }
        }

        Ok(current_state)
    }

    /// Get the next node based on edges
    fn get_next_node(&self, current: &str, state: &S) -> Result<Option<String>> {
        let edges = match self.structure.edges.get(current) {
            Some(e) => e,
            None => return Ok(None),
        };

        for edge in edges {
            match edge {
                EdgeType::Direct(to) => {
                    return Ok(Some(to.clone()));
                }
                EdgeType::Conditional { condition, edge_map } => {
                    let key = condition(state);
                    if let Some(to) = edge_map.get(&key) {
                        return Ok(Some(to.clone()));
                    }
                }
            }
        }

        Ok(None)
    }

    /// Get all nodes in the graph
    pub fn nodes(&self) -> Vec<String> {
        self.structure.nodes.keys().cloned().collect()
    }

    /// Get the entry point
    pub fn entry_point(&self) -> Option<&str> {
        self.structure.entry_point.as_deref()
    }

    /// Get the finish point
    pub fn finish_point(&self) -> Option<&str> {
        self.structure.finish_point.as_deref()
    }
}

/// Configuration for graph execution
#[derive(Debug, Clone)]
pub struct ExecutionConfig {
    /// Maximum number of steps to execute
    pub max_steps: Option<usize>,
    /// Enable debug logging
    pub debug: bool,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            max_steps: Some(1000),
            debug: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::FunctionNode;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct TestState {
        value: i32,
    }

    #[tokio::test]
    async fn test_simple_graph() {
        let graph = StateGraph::new()
            .add_node(
                "increment",
                FunctionNode::new("increment", |mut state: TestState| {
                    Box::pin(async move {
                        state.value += 1;
                        Ok(state)
                    })
                }),
            )
            .add_node(
                "double",
                FunctionNode::new("double", |mut state: TestState| {
                    Box::pin(async move {
                        state.value *= 2;
                        Ok(state)
                    })
                }),
            )
            .add_edge("increment", "double")
            .set_entry_point("increment")
            .set_finish_point("double");

        let compiled = graph.compile().unwrap();
        let state = TestState { value: 5 };
        let result = compiled.execute(state).await.unwrap();

        assert_eq!(result.value, 12); // (5 + 1) * 2
    }

    #[tokio::test]
    async fn test_conditional_graph() {
        let mut edge_map = HashMap::new();
        edge_map.insert("increment".to_string(), "increment".to_string());
        edge_map.insert("done".to_string(), "finish".to_string());

        let graph = StateGraph::new()
            .add_node(
                "increment",
                FunctionNode::new("increment", |mut state: TestState| {
                    Box::pin(async move {
                        state.value += 1;
                        Ok(state)
                    })
                }),
            )
            .add_node(
                "finish",
                FunctionNode::new("finish", |state: TestState| {
                    Box::pin(async move { Ok(state) })
                }),
            )
            .add_conditional_edge(
                "increment",
                |state: &TestState| {
                    if state.value < 10 {
                        "increment".to_string()
                    } else {
                        "done".to_string()
                    }
                },
                edge_map,
            )
            .set_entry_point("increment")
            .set_finish_point("finish");

        let compiled = graph.compile().unwrap();
        let state = TestState { value: 0 };
        let result = compiled.execute(state).await.unwrap();

        assert_eq!(result.value, 10);
    }
}
