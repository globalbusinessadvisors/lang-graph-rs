//! Graph execution engine - StateGraph and CompiledGraph

use crate::interactive::{
    ExecutionEvent, ExecutionHandle, InteractionPosition, InteractionResponse, InteractiveConfig,
    InteractiveState, InterruptReason,
};
use crate::{Error, Node, Result, State};
use dashmap::DashMap;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};

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

    /// Execute the graph with interactive human-in-the-loop support
    ///
    /// This method allows execution to be paused at specified breakpoints,
    /// enabling human intervention to inspect state, modify it, or control flow.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use langgraph_core::interactive::InteractiveConfig;
    ///
    /// let config = InteractiveConfig::with_breakpoints_before(vec!["review_node"]);
    /// let (handle, result_rx) = graph.execute_interactive(state, config).await?;
    ///
    /// // In a separate task or thread
    /// tokio::spawn(async move {
    ///     while let Some((point, state)) = handle.wait_for_interaction().await {
    ///         println!("Paused at: {:?}", point);
    ///         // Modify state or just continue
    ///         handle.continue_execution().unwrap();
    ///     }
    /// });
    ///
    /// let final_state = result_rx.await.unwrap()?;
    /// ```
    pub async fn execute_interactive(
        &self,
        state: S,
        config: InteractiveConfig,
    ) -> Result<(ExecutionHandle<S>, tokio::task::JoinHandle<Result<S>>)> {
        let (interaction_tx, interaction_rx) = mpsc::unbounded_channel();
        let (response_tx, response_rx) = mpsc::unbounded_channel();
        let trace = Arc::new(RwLock::new(crate::interactive::ExecutionTrace::new()));

        let handle = ExecutionHandle::new(interaction_rx, response_tx, trace.clone());

        let mut interactive_state = InteractiveState {
            interaction_tx,
            response_rx,
            trace: trace.clone(),
            config: config.clone(),
            step_count: 0,
        };

        let self_clone = self.clone();
        let execution_task = tokio::spawn(async move {
            self_clone
                .execute_interactive_impl(state, &mut interactive_state)
                .await
        });

        Ok((handle, execution_task))
    }

    /// Internal implementation of interactive execution
    async fn execute_interactive_impl(
        &self,
        mut state: S,
        interactive: &mut InteractiveState<S>,
    ) -> Result<S> {
        let entry = self
            .structure
            .entry_point
            .as_ref()
            .ok_or_else(|| Error::invalid_graph("No entry point"))?;

        let mut current = entry.clone();
        let mut visited = HashSet::new();

        loop {
            // Check max steps
            if let Some(max_steps) = interactive.config.max_steps {
                if interactive.step_count >= max_steps {
                    return Err(Error::invalid_operation(format!(
                        "Max steps ({}) exceeded",
                        max_steps
                    )));
                }
            }

            visited.insert(current.clone());
            interactive.step_count += 1;

            // Check for interruption before node execution
            if let Some(reason) =
                interactive.should_interrupt(&current, InteractionPosition::Before)
            {
                let response = interactive
                    .interact(
                        current.clone(),
                        InteractionPosition::Before,
                        state.clone(),
                        reason,
                    )
                    .await?;

                match response {
                    InteractionResponse::Continue => {}
                    InteractionResponse::ContinueWith(new_state) => {
                        state = new_state;
                    }
                    InteractionResponse::Skip => {
                        // Skip this node, move to next
                        if let Some(next) = self.get_next_node(&current, &state)? {
                            current = next;
                            continue;
                        } else {
                            break;
                        }
                    }
                    InteractionResponse::Abort(reason) => {
                        return Err(Error::invalid_operation(format!(
                            "Execution aborted: {}",
                            reason
                        )));
                    }
                    InteractionResponse::ResumeAt(node) => {
                        current = node;
                        continue;
                    }
                }
            }

            // Get and execute the node
            let node = self
                .structure
                .nodes
                .get(&current)
                .ok_or_else(|| Error::node_not_found(&current))?;

            let node_name = current.clone();
            let execution_result = node.execute(state.clone()).await;

            // Record execution event
            let event = match &execution_result {
                Ok(_) => ExecutionEvent {
                    node_name: node_name.clone(),
                    step: interactive.step_count,
                    timestamp: std::time::SystemTime::now(),
                    success: true,
                    error: None,
                },
                Err(e) => ExecutionEvent {
                    node_name: node_name.clone(),
                    step: interactive.step_count,
                    timestamp: std::time::SystemTime::now(),
                    success: false,
                    error: Some(e.to_string()),
                },
            };

            interactive.record_event(event).await;

            // Handle execution result
            match execution_result {
                Ok(new_state) => {
                    state = new_state;
                }
                Err(e) => {
                    // Check if we should interrupt on error
                    if interactive
                        .config
                        .strategies
                        .iter()
                        .any(|s| matches!(s, crate::interactive::InterruptStrategy::OnError))
                    {
                        let response = interactive
                            .interact(
                                node_name.clone(),
                                InteractionPosition::After,
                                state.clone(),
                                InterruptReason::Error(e.to_string()),
                            )
                            .await?;

                        match response {
                            InteractionResponse::Continue => {
                                // Continue despite error
                            }
                            InteractionResponse::ContinueWith(new_state) => {
                                state = new_state;
                            }
                            InteractionResponse::Abort(reason) => {
                                return Err(Error::invalid_operation(format!(
                                    "Execution aborted: {}",
                                    reason
                                )));
                            }
                            _ => {
                                return Err(e);
                            }
                        }
                    } else {
                        return Err(Error::node_execution_failed(format!(
                            "Node '{}': {}",
                            node_name, e
                        )));
                    }
                }
            }

            // Check for interruption after node execution
            if let Some(reason) = interactive.should_interrupt(&current, InteractionPosition::After)
            {
                let response = interactive
                    .interact(current.clone(), InteractionPosition::After, state.clone(), reason)
                    .await?;

                match response {
                    InteractionResponse::Continue => {}
                    InteractionResponse::ContinueWith(new_state) => {
                        state = new_state;
                    }
                    InteractionResponse::Skip => {}
                    InteractionResponse::Abort(reason) => {
                        return Err(Error::invalid_operation(format!(
                            "Execution aborted: {}",
                            reason
                        )));
                    }
                    InteractionResponse::ResumeAt(node) => {
                        current = node;
                        continue;
                    }
                }
            }

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
                    break;
                }
            }
        }

        // Mark trace as complete
        let mut trace = interactive.trace.write().await;
        trace.complete();

        Ok(state)
    }
}

impl<S: State> Clone for CompiledGraph<S> {
    fn clone(&self) -> Self {
        Self {
            structure: self.structure.clone(),
            execution_cache: self.execution_cache.clone(),
        }
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
