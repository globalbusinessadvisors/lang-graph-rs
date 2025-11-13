//! Integration tests for enhanced features: streaming, HITL, and time travel

use futures::StreamExt;
use langgraph_checkpoint::{Checkpoint, Checkpointer};
use langgraph_core::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct TestState {
    count: i32,
    message: String,
    history: Vec<String>,
}

/// Simple in-memory checkpointer for testing
struct MemoryCheckpointer<S: State> {
    checkpoints: Arc<RwLock<HashMap<String, Checkpoint<S>>>>,
}

impl<S: State> MemoryCheckpointer<S> {
    fn new() -> Self {
        Self {
            checkpoints: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

#[async_trait::async_trait]
impl<S: State> Checkpointer<S> for MemoryCheckpointer<S> {
    async fn save(&self, checkpoint: Checkpoint<S>) -> langgraph_core::Result<String> {
        let id = checkpoint.metadata.id.clone();
        self.checkpoints.write().await.insert(id.clone(), checkpoint);
        Ok(id)
    }

    async fn load(&self, checkpoint_id: &str) -> langgraph_core::Result<Option<Checkpoint<S>>> {
        Ok(self.checkpoints.read().await.get(checkpoint_id).cloned())
    }

    async fn load_latest(&self, thread_id: &str) -> langgraph_core::Result<Option<Checkpoint<S>>> {
        let checkpoints = self.checkpoints.read().await;
        let mut latest: Option<Checkpoint<S>> = None;

        for checkpoint in checkpoints.values() {
            if checkpoint.metadata.thread_id == thread_id {
                if latest.is_none()
                    || checkpoint.metadata.created_at > latest.as_ref().unwrap().metadata.created_at
                {
                    latest = Some(checkpoint.clone());
                }
            }
        }

        Ok(latest)
    }

    async fn list(
        &self,
        thread_id: &str,
    ) -> langgraph_core::Result<Vec<langgraph_checkpoint::CheckpointMetadata>> {
        let checkpoints = self.checkpoints.read().await;
        let mut metadata: Vec<_> = checkpoints
            .values()
            .filter(|cp| cp.metadata.thread_id == thread_id)
            .map(|cp| cp.metadata.clone())
            .collect();

        metadata.sort_by(|a, b| a.created_at.cmp(&b.created_at));
        Ok(metadata)
    }

    async fn list_paginated(
        &self,
        thread_id: &str,
        limit: usize,
        offset: usize,
    ) -> langgraph_core::Result<Vec<langgraph_checkpoint::CheckpointMetadata>> {
        let all = self.list(thread_id).await?;
        Ok(all.into_iter().skip(offset).take(limit).collect())
    }

    async fn delete(&self, checkpoint_id: &str) -> langgraph_core::Result<bool> {
        Ok(self.checkpoints.write().await.remove(checkpoint_id).is_some())
    }

    async fn delete_thread(&self, thread_id: &str) -> langgraph_core::Result<usize> {
        let mut checkpoints = self.checkpoints.write().await;
        let to_remove: Vec<_> = checkpoints
            .iter()
            .filter(|(_, cp)| cp.metadata.thread_id == thread_id)
            .map(|(id, _)| id.clone())
            .collect();

        let count = to_remove.len();
        for id in to_remove {
            checkpoints.remove(&id);
        }

        Ok(count)
    }

    async fn count(&self, thread_id: &str) -> langgraph_core::Result<usize> {
        Ok(self
            .checkpoints
            .read()
            .await
            .values()
            .filter(|cp| cp.metadata.thread_id == thread_id)
            .count())
    }

    async fn search(
        &self,
        thread_id: &str,
        _metadata_filter: HashMap<String, serde_json::Value>,
    ) -> langgraph_core::Result<Vec<langgraph_checkpoint::CheckpointMetadata>> {
        // Simplified search - just return all for thread
        self.list(thread_id).await
    }
}


// Create a newtype wrapper to satisfy orphan rules
struct CheckpointMetadataWrapper(langgraph_checkpoint::CheckpointMetadata);

impl langgraph_core::CheckpointMetadata for CheckpointMetadataWrapper {
    fn id(&self) -> &str {
        &self.0.id
    }

    fn created_at(&self) -> chrono::DateTime<chrono::Utc> {
        self.0.created_at
    }
}

// Update MemoryCheckpointer to use wrapper
struct MemoryCheckpointerWrapper<S: State> {
    checkpoints: Arc<RwLock<HashMap<String, Checkpoint<S>>>>,
}

impl<S: State> MemoryCheckpointerWrapper<S> {
    fn new() -> Self {
        Self {
            checkpoints: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    fn from_checkpointer(cp: &MemoryCheckpointer<S>) -> Self {
        Self {
            checkpoints: cp.checkpoints.clone(),
        }
    }
}

#[async_trait::async_trait]
impl<S: State> langgraph_core::CheckpointLoader<S> for MemoryCheckpointerWrapper<S> {
    type Metadata = CheckpointMetadataWrapper;

    async fn load_checkpoint(&self, checkpoint_id: &str) -> langgraph_core::Result<Option<langgraph_core::CheckpointData<S>>> {
        let checkpoints = self.checkpoints.read().await;
        if let Some(cp) = checkpoints.get(checkpoint_id) {
            Ok(Some(langgraph_core::CheckpointData::new(
                cp.metadata.id.clone(),
                cp.state.clone(),
                cp.metadata.created_at,
            )))
        } else {
            Ok(None)
        }
    }

    async fn list_metadata(&self, thread_id: &str) -> langgraph_core::Result<Vec<Self::Metadata>> {
        let checkpoints = self.checkpoints.read().await;
        let mut metadata: Vec<_> = checkpoints
            .values()
            .filter(|cp| cp.metadata.thread_id == thread_id)
            .map(|cp| CheckpointMetadataWrapper(cp.metadata.clone()))
            .collect();

        metadata.sort_by(|a, b| a.0.created_at.cmp(&b.0.created_at));
        Ok(metadata)
    }
}

#[tokio::test]
async fn test_enhanced_streaming() {
    use langgraph_core::node::FunctionNode;

    let graph = StateGraph::new()
        .add_node(
            "start",
            FunctionNode::new("start", |mut state: TestState| {
                Box::pin(async move {
                    state.count += 1;
                    state.history.push("start".to_string());
                    Ok(state)
                })
            }),
        )
        .add_node(
            "middle",
            FunctionNode::new("middle", |mut state: TestState| {
                Box::pin(async move {
                    state.count *= 2;
                    state.history.push("middle".to_string());
                    Ok(state)
                })
            }),
        )
        .add_node(
            "end",
            FunctionNode::new("end", |mut state: TestState| {
                Box::pin(async move {
                    state.message = format!("Final count: {}", state.count);
                    state.history.push("end".to_string());
                    Ok(state)
                })
            }),
        )
        .add_edge("start", "middle")
        .add_edge("middle", "end")
        .set_entry_point("start")
        .set_finish_point("end");

    let compiled = graph.compile().unwrap();

    let initial_state = TestState {
        count: 0,
        message: String::new(),
        history: Vec::new(),
    };

    let config = StreamConfig::new()
        .with_max_steps(10)
        .with_checkpoints(true)
        .with_metadata(true);

    let mut stream = compiled.stream_with_config(initial_state, config).await.unwrap();

    let mut events = Vec::new();
    while let Some(event_result) = stream.next().await {
        let event = event_result.unwrap();
        println!("Event: {:?}", event);
        events.push(event);
    }

    // Should have: Started, NodeComplete (start), NodeComplete (middle), Completed
    assert!(events.len() >= 4, "Expected at least 4 events, got {}", events.len());

    // Check first event is Started
    assert!(matches!(&events[0], StreamEvent::Started { .. }));

    // Check last event is Completed
    let last = events.last().unwrap();
    if let StreamEvent::Completed {
        final_node,
        state,
        total_steps,
        ..
    } = last
    {
        assert_eq!(final_node, "end");
        assert_eq!(state.count, 2); // (0 + 1) * 2
        assert_eq!(state.history, vec!["start", "middle", "end"]);
        assert_eq!(*total_steps, 3);
    } else {
        panic!("Last event should be Completed");
    }
}

#[tokio::test]
async fn test_human_in_the_loop() {
    use langgraph_core::node::FunctionNode;

    let graph = StateGraph::new()
        .add_node(
            "start",
            FunctionNode::new("start", |mut state: TestState| {
                Box::pin(async move {
                    state.count += 1;
                    state.history.push("start".to_string());
                    Ok(state)
                })
            }),
        )
        .add_node(
            "needs_approval",
            FunctionNode::new("needs_approval", |mut state: TestState| {
                Box::pin(async move {
                    state.count *= 2;
                    state.history.push("needs_approval".to_string());
                    Ok(state)
                })
            }),
        )
        .add_node(
            "end",
            FunctionNode::new("end", |mut state: TestState| {
                Box::pin(async move {
                    state.message = format!("Approved: {}", state.count);
                    state.history.push("end".to_string());
                    Ok(state)
                })
            }),
        )
        .add_edge("start", "needs_approval")
        .add_edge("needs_approval", "end")
        .set_entry_point("start")
        .set_finish_point("end");

    let compiled = graph.compile().unwrap();

    let approval_config = ApprovalConfig::new(
        "Approve doubling the count?",
        vec!["Yes".to_string(), "No".to_string()],
    )
    .with_default("Yes");

    let interrupt_config = InterruptConfig::new()
        .require_approval("needs_approval", approval_config);

    let (handler, mut interrupt_rx, response_tx) = InterruptHandler::new(interrupt_config);

    let initial_state = TestState {
        count: 5,
        message: String::new(),
        history: Vec::new(),
    };

    // Spawn execution in background
    let compiled_clone = compiled.clone();
    let handler_clone = handler.clone();
    let execution_handle = tokio::spawn(async move {
        compiled_clone
            .execute_with_interrupt(initial_state, ExecutionConfig::default(), handler_clone)
            .await
    });

    // Wait for interrupt
    if let Some(interrupt_event) = interrupt_rx.recv().await {
        assert_eq!(interrupt_event.node_name, "needs_approval");
        assert!(matches!(
            interrupt_event.signal,
            InterruptSignal::ApprovalRequired { .. }
        ));

        // Send approval
        response_tx
            .send(InterruptResponse::Approved {
                selected_option: Some("Yes".to_string()),
                data: HashMap::new(),
            })
            .await
            .unwrap();
    } else {
        panic!("Expected interrupt event");
    }

    // Wait for execution to complete
    let result = execution_handle.await.unwrap().unwrap();
    assert_eq!(result.count, 12); // (5 + 1) * 2
    assert_eq!(result.history, vec!["start", "needs_approval", "end"]);
}

#[tokio::test]
async fn test_time_travel_debugging() {
    use langgraph_core::node::FunctionNode;

    let checkpointer = MemoryCheckpointer::new();
    let checkpointer_wrapper = MemoryCheckpointerWrapper::from_checkpointer(&checkpointer);

    let graph = StateGraph::new()
        .add_node(
            "step1",
            FunctionNode::new("step1", |mut state: TestState| {
                Box::pin(async move {
                    state.count += 10;
                    state.history.push("step1".to_string());
                    Ok(state)
                })
            }),
        )
        .add_node(
            "step2",
            FunctionNode::new("step2", |mut state: TestState| {
                Box::pin(async move {
                    state.count += 20;
                    state.history.push("step2".to_string());
                    Ok(state)
                })
            }),
        )
        .add_node(
            "step3",
            FunctionNode::new("step3", |mut state: TestState| {
                Box::pin(async move {
                    state.count += 30;
                    state.history.push("step3".to_string());
                    Ok(state)
                })
            }),
        )
        .add_edge("step1", "step2")
        .add_edge("step2", "step3")
        .set_entry_point("step1")
        .set_finish_point("step3");

    let compiled = graph.compile().unwrap();

    // Execute and save checkpoints
    let state1 = TestState {
        count: 0,
        message: String::new(),
        history: Vec::new(),
    };

    // Manually simulate execution with checkpoints
    let state2 = TestState {
        count: 10,
        message: String::new(),
        history: vec!["step1".to_string()],
    };

    let state3 = TestState {
        count: 30,
        message: String::new(),
        history: vec!["step1".to_string(), "step2".to_string()],
    };

    let cp1 = Checkpoint::new("thread-1", state1.clone());
    let cp2 = Checkpoint::with_parent("thread-1", cp1.metadata.id.clone(), state2.clone());
    let cp3 = Checkpoint::with_parent("thread-1", cp2.metadata.id.clone(), state3.clone());

    let cp1_id = checkpointer.save(cp1).await.unwrap();
    let cp2_id = checkpointer.save(cp2).await.unwrap();
    let _cp3_id = checkpointer.save(cp3).await.unwrap();

    // Test state diffing - use the wrapper (which shares the same storage)
    let comparison = TimeTravelDebugger::compare_checkpoints(&checkpointer_wrapper, &cp1_id, &cp2_id)
        .await
        .unwrap();

    assert!(comparison.diff.has_changes());
    assert_eq!(comparison.first_state.count, 0);
    assert_eq!(comparison.second_state.count, 10);

    // Test replay
    let replay_config = ReplayConfig::new().with_max_steps(10);
    let replayed_state = compiled
        .replay_from_checkpoint(&checkpointer_wrapper, &cp2_id, replay_config)
        .await
        .unwrap();

    // After replay from cp2 (which has count=10), it re-executes from entry point
    // So: 10 (from cp2) + 10 (step1) + 20 (step2) + 30 (step3) = 70
    assert_eq!(replayed_state.count, 70);

    // Test timeline
    let timeline = TimeTravelDebugger::build_timeline(&checkpointer_wrapper, "thread-1")
        .await
        .unwrap();

    assert_eq!(timeline.len(), 3);
    assert_eq!(timeline[0].state.count, 0);
    assert_eq!(timeline[1].state.count, 10);
    assert_eq!(timeline[2].state.count, 30);
}

#[tokio::test]
async fn test_state_diffing() {
    let state1 = TestState {
        count: 5,
        message: "Hello".to_string(),
        history: vec!["a".to_string(), "b".to_string()],
    };

    let state2 = TestState {
        count: 10,
        message: "Hello".to_string(),
        history: vec!["a".to_string(), "b".to_string(), "c".to_string()],
    };

    let diff = StateDiffer::diff("cp1", "cp2", &state1, &state2).unwrap();

    assert!(diff.has_changes());
    println!("Diff changes: {}", diff.total_changes());
    println!("Modified: {:?}", diff.modified);
    println!("Added: {:?}", diff.added);

    // Should detect count change and history array change
    assert!(diff.total_changes() > 0);
}

#[tokio::test]
async fn test_conditional_with_streaming() {
    use langgraph_core::node::FunctionNode;

    let mut edge_map = HashMap::new();
    edge_map.insert("continue".to_string(), "increment".to_string());
    edge_map.insert("done".to_string(), "finish".to_string());

    let graph = StateGraph::new()
        .add_node(
            "increment",
            FunctionNode::new("increment", |mut state: TestState| {
                Box::pin(async move {
                    state.count += 1;
                    state.history.push(format!("increment:{}", state.count));
                    Ok(state)
                })
            }),
        )
        .add_node(
            "finish",
            FunctionNode::new("finish", |mut state: TestState| {
                Box::pin(async move {
                    state.message = "Done!".to_string();
                    state.history.push("finish".to_string());
                    Ok(state)
                })
            }),
        )
        .add_conditional_edge(
            "increment",
            |state: &TestState| {
                if state.count < 3 {
                    "continue".to_string()
                } else {
                    "done".to_string()
                }
            },
            edge_map,
        )
        .set_entry_point("increment")
        .set_finish_point("finish");

    let compiled = graph.compile().unwrap();

    let initial_state = TestState {
        count: 0,
        message: String::new(),
        history: Vec::new(),
    };

    let config = StreamConfig::new().with_max_steps(10);

    let mut stream = compiled.stream_with_config(initial_state, config).await.unwrap();

    let mut final_state = None;
    while let Some(event_result) = stream.next().await {
        let event = event_result.unwrap();
        if let StreamEvent::Completed { state, .. } = event {
            final_state = Some(state);
        }
    }

    let final_state = final_state.expect("Should have completed");
    assert_eq!(final_state.count, 3);
    assert_eq!(final_state.message, "Done!");
    assert!(final_state.history.contains(&"finish".to_string()));
}
