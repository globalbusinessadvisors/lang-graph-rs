//! Node trait and implementations

use crate::{Error, Result, State};
use async_trait::async_trait;
use std::fmt::Debug;
use std::sync::Arc;

/// Async node execution interface
///
/// Nodes represent computational units in the graph that transform state.
#[async_trait]
pub trait Node<S>: Send + Sync + Debug
where
    S: State,
{
    /// Execute the node with the given state
    ///
    /// # Arguments
    /// * `state` - Current state to process
    ///
    /// # Returns
    /// Updated state after node execution
    async fn execute(&self, state: S) -> Result<S>;

    /// Get the name of this node
    fn name(&self) -> &str;

    /// Optional validation before execution
    async fn validate(&self, _state: &S) -> Result<()> {
        Ok(())
    }
}

/// Type alias for boxed node trait objects
pub type BoxedNode<S> = Arc<dyn Node<S>>;

/// Function-based node implementation
#[derive(Clone)]
pub struct FunctionNode<S, F>
where
    S: State,
    F: Fn(S) -> futures::future::BoxFuture<'static, Result<S>> + Send + Sync + 'static,
{
    name: String,
    function: Arc<F>,
    _phantom: std::marker::PhantomData<S>,
}

impl<S, F> FunctionNode<S, F>
where
    S: State,
    F: Fn(S) -> futures::future::BoxFuture<'static, Result<S>> + Send + Sync + 'static,
{
    /// Create a new function node
    pub fn new(name: impl Into<String>, function: F) -> Self {
        Self {
            name: name.into(),
            function: Arc::new(function),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<S, F> Debug for FunctionNode<S, F>
where
    S: State,
    F: Fn(S) -> futures::future::BoxFuture<'static, Result<S>> + Send + Sync + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FunctionNode")
            .field("name", &self.name)
            .finish()
    }
}

#[async_trait]
impl<S, F> Node<S> for FunctionNode<S, F>
where
    S: State + 'static,
    F: Fn(S) -> futures::future::BoxFuture<'static, Result<S>> + Send + Sync + 'static,
{
    async fn execute(&self, state: S) -> Result<S> {
        (self.function)(state).await
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Conditional node that executes based on state condition
pub struct ConditionalNode<S, C, T, F>
where
    S: State,
    C: Fn(&S) -> bool + Send + Sync + 'static,
    T: Node<S>,
    F: Node<S>,
{
    name: String,
    condition: Arc<C>,
    true_branch: Arc<T>,
    false_branch: Arc<F>,
    _phantom: std::marker::PhantomData<S>,
}

impl<S, C, T, F> ConditionalNode<S, C, T, F>
where
    S: State,
    C: Fn(&S) -> bool + Send + Sync + 'static,
    T: Node<S>,
    F: Node<S>,
{
    /// Create a new conditional node
    pub fn new(name: impl Into<String>, condition: C, true_branch: T, false_branch: F) -> Self {
        Self {
            name: name.into(),
            condition: Arc::new(condition),
            true_branch: Arc::new(true_branch),
            false_branch: Arc::new(false_branch),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<S, C, T, F> Debug for ConditionalNode<S, C, T, F>
where
    S: State,
    C: Fn(&S) -> bool + Send + Sync + 'static,
    T: Node<S>,
    F: Node<S>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConditionalNode")
            .field("name", &self.name)
            .field("true_branch", &self.true_branch)
            .field("false_branch", &self.false_branch)
            .finish()
    }
}

#[async_trait]
impl<S, C, T, F> Node<S> for ConditionalNode<S, C, T, F>
where
    S: State + 'static,
    C: Fn(&S) -> bool + Send + Sync + 'static,
    T: Node<S> + 'static,
    F: Node<S> + 'static,
{
    async fn execute(&self, state: S) -> Result<S> {
        if (self.condition)(&state) {
            self.true_branch.execute(state).await
        } else {
            self.false_branch.execute(state).await
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct TestState {
        count: i32,
    }

    #[tokio::test]
    async fn test_function_node() {
        let node = FunctionNode::new("increment", |mut state: TestState| {
            Box::pin(async move {
                state.count += 1;
                Ok(state)
            })
        });

        let state = TestState { count: 0 };
        let result = node.execute(state).await.unwrap();
        assert_eq!(result.count, 1);
    }
}
