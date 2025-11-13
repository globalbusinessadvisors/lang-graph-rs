//! Graph execution engine - StateGraph and CompiledGraph

use crate::interrupt::{Interrupt, InterruptManager, InterruptReason, InterruptResponse};
use crate::timetravel::{ExecutionHistoryManager, ExecutionStep};
use crate::{Error, Node, Result, State};
use dashmap::DashMap;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

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

/// Internal graph structure
struct GraphStructure<S: State> {
    nodes: HashMap<String, Arc<dyn Node<S>>>,
    edges: HashMap<String, Vec<EdgeType<S>>>,
    entry_point: Option<String>,
    finish_point: Option<String>,
}

/// StateGraph builder for creating execution graphs
pub struct StateGraph<S: State> {
    nodes: HashMap<String, Arc<dyn Node<S>>>,
    edges: HashMap<String, Vec<EdgeType<S>>>,
    entry_point: Option<String>,
    finish_point: Option<String>,
}

impl<S: State> StateGraph<S> {
    /// Create a new StateGraph
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            entry_point: None,
            finish_point: None,
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

    /// Compile the graph into an executable form
    pub fn compile(self) -> Result<CompiledGraph<S>> {
        // Validate graph structure
        self.validate()?;

        let structure = Arc::new(GraphStructure {
            nodes: self.nodes,
            edges: self.edges,
            entry_point: self.entry_point,
            finish_point: self.finish_point,
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
#[derive(Clone)]
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
        config: ExecutionConfig<S>,
    ) -> Result<S> {
        let entry = self
            .structure
            .entry_point
            .as_ref()
            .ok_or_else(|| Error::invalid_graph("No entry point"))?;

        // Start execution history if time travel is enabled
        if config.enable_time_travel {
            if let Some(ref history_manager) = config.history_manager {
                history_manager
                    .start_execution(&config.thread_id, state.clone())
                    .await;
            }
        }

        let mut current = entry.clone();
        let mut visited = HashSet::new();
        let mut step_count = 0;

        let execution_result = async {
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

                // Check for interrupt before execution
                if let Some(reason) = config.interrupt_nodes.get(&current) {
                    if let Some(ref interrupt_manager) = config.interrupt_manager {
                        let interrupt = Interrupt::new(
                            config.thread_id.clone(),
                            current.clone(),
                            state.clone(),
                            reason.clone(),
                        );

                        if config.debug {
                            eprintln!("[DEBUG] Interrupt triggered at node: {}", current);
                        }

                        let response = interrupt_manager.register_interrupt(interrupt).await?;

                        if !response.should_continue {
                            return Err(Error::invalid_operation(
                                "Execution aborted by human interrupt",
                            ));
                        }

                        // Apply state updates if provided
                        if let Some(updates) = response.state_updates {
                            // For now, we'll just log this - actual state merging would require
                            // reflection or a more sophisticated state update mechanism
                            if config.debug {
                                eprintln!("[DEBUG] State updates requested: {:?}", updates);
                            }
                        }
                    }
                }

                // Get the node
                let node = self
                    .structure
                    .nodes
                    .get(&current)
                    .ok_or_else(|| Error::node_not_found(&current))?;

                // Store state before execution for history
                let state_before = state.clone();
                let start_time = std::time::Instant::now();

                // Execute the node
                let execution_result = node.execute(state).await;
                let duration = start_time.elapsed();

                match execution_result {
                    Ok(new_state) => {
                        state = new_state;

                        // Record step in history
                        if config.enable_time_travel {
                            if let Some(ref history_manager) = config.history_manager {
                                let step = ExecutionStep::new(
                                    step_count,
                                    current.clone(),
                                    state_before,
                                    state.clone(),
                                    duration.as_micros() as u64,
                                );
                                let _ = history_manager
                                    .add_step(&config.thread_id, step)
                                    .await;
                            }
                        }

                        if config.debug {
                            eprintln!(
                                "[DEBUG] Node '{}' completed in {:?}",
                                current, duration
                            );
                        }
                    }
                    Err(e) => {
                        // Record error step in history
                        if config.enable_time_travel {
                            if let Some(ref history_manager) = config.history_manager {
                                let step = ExecutionStep::with_error(
                                    step_count,
                                    current.clone(),
                                    state_before,
                                    e.to_string(),
                                );
                                let _ = history_manager
                                    .add_step(&config.thread_id, step)
                                    .await;
                            }
                        }

                        return Err(Error::node_execution_failed(format!(
                            "Node '{}': {}",
                            current, e
                        )));
                    }
                }

                step_count += 1;

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
        .await;

        // Complete execution history
        if config.enable_time_travel {
            if let Some(ref history_manager) = config.history_manager {
                let _ = history_manager
                    .complete_execution(&config.thread_id, execution_result.is_ok())
                    .await;
            }
        }

        execution_result
    }

    /// Stream execution with intermediate states
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
#[derive(Clone)]
pub struct ExecutionConfig<S: State> {
    /// Maximum number of steps to execute
    pub max_steps: Option<usize>,
    /// Enable debug logging
    pub debug: bool,
    /// Thread ID for this execution
    pub thread_id: String,
    /// Interrupt manager for human-in-the-loop
    pub interrupt_manager: Option<Arc<InterruptManager<S>>>,
    /// Nodes that should trigger interrupts (node_name -> when to interrupt)
    pub interrupt_nodes: HashMap<String, InterruptReason>,
    /// History manager for time travel debugging
    pub history_manager: Option<Arc<ExecutionHistoryManager<S>>>,
    /// Enable automatic checkpointing at each step
    pub enable_time_travel: bool,
}

impl<S: State> ExecutionConfig<S> {
    /// Create a new configuration with thread ID
    pub fn new(thread_id: impl Into<String>) -> Self {
        Self {
            max_steps: Some(1000),
            debug: false,
            thread_id: thread_id.into(),
            interrupt_manager: None,
            interrupt_nodes: HashMap::new(),
            history_manager: None,
            enable_time_travel: false,
        }
    }

    /// Enable human-in-the-loop with the provided interrupt manager
    pub fn with_interrupts(mut self, manager: Arc<InterruptManager<S>>) -> Self {
        self.interrupt_manager = Some(manager);
        self
    }

    /// Add a node that should trigger an interrupt
    pub fn add_interrupt_node(
        mut self,
        node_name: impl Into<String>,
        reason: InterruptReason,
    ) -> Self {
        self.interrupt_nodes.insert(node_name.into(), reason);
        self
    }

    /// Enable time travel debugging with the provided history manager
    pub fn with_time_travel(mut self, manager: Arc<ExecutionHistoryManager<S>>) -> Self {
        self.history_manager = Some(manager);
        self.enable_time_travel = true;
        self
    }

    /// Enable time travel debugging with a new history manager
    pub fn enable_time_travel(mut self) -> Self {
        self.history_manager = Some(Arc::new(ExecutionHistoryManager::new()));
        self.enable_time_travel = true;
        self
    }

    /// Set maximum steps
    pub fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = Some(max_steps);
        self
    }

    /// Enable debug logging
    pub fn with_debug(mut self, debug: bool) -> Self {
        self.debug = debug;
        self
    }
}

impl<S: State> Default for ExecutionConfig<S> {
    fn default() -> Self {
        Self {
            max_steps: Some(1000),
            debug: false,
            thread_id: uuid::Uuid::new_v4().to_string(),
            interrupt_manager: None,
            interrupt_nodes: HashMap::new(),
            history_manager: None,
            enable_time_travel: false,
        }
    }
}

impl<S: State> std::fmt::Debug for ExecutionConfig<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExecutionConfig")
            .field("max_steps", &self.max_steps)
            .field("debug", &self.debug)
            .field("thread_id", &self.thread_id)
            .field("has_interrupt_manager", &self.interrupt_manager.is_some())
            .field("interrupt_nodes", &self.interrupt_nodes)
            .field("has_history_manager", &self.history_manager.is_some())
            .field("enable_time_travel", &self.enable_time_travel)
            .finish()
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
