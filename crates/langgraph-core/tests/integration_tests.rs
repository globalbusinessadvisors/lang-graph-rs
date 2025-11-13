//! Integration tests for Human-in-the-loop and Time travel debugging features

use langgraph_core::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct TestState {
    counter: i32,
    message: String,
}

#[tokio::test]
async fn test_human_in_the_loop_approval() {
    let interrupt_manager = Arc::new(InterruptManager::<TestState>::new());

    // Build a graph with an interrupt node
    let graph = StateGraph::new()
        .add_node(
            "increment",
            FunctionNode::new("increment", |mut state: TestState| {
                Box::pin(async move {
                    state.counter += 1;
                    state.message = format!("Count: {}", state.counter);
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
        .add_edge("increment", "finish")
        .set_entry_point("increment")
        .set_finish_point("finish")
        .compile()
        .unwrap();

    let config = ExecutionConfig::new("test-thread-1")
        .with_interrupts(interrupt_manager.clone())
        .add_interrupt_node("increment", InterruptReason::ApprovalRequired);

    let initial_state = TestState {
        counter: 0,
        message: String::new(),
    };

    // Execute in background
    let graph_clone = graph.clone();
    let execute_handle = tokio::spawn(async move {
        graph_clone
            .execute_with_config(initial_state, config)
            .await
    });

    // Wait for interrupt
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

    // Get active interrupts
    let interrupts = interrupt_manager.get_active_interrupts().await;
    assert_eq!(interrupts.len(), 1);
    assert_eq!(interrupts[0].node_name, "increment");

    // Approve the interrupt
    let response = InterruptResponse::approve(&interrupts[0].id);
    interrupt_manager.respond(response).await.unwrap();

    // Wait for execution to complete
    let result = execute_handle.await.unwrap().unwrap();
    assert_eq!(result.counter, 1);
    assert_eq!(result.message, "Count: 1");
}

#[tokio::test]
async fn test_human_in_the_loop_rejection() {
    let interrupt_manager = Arc::new(InterruptManager::<TestState>::new());

    let graph = StateGraph::new()
        .add_node(
            "increment",
            FunctionNode::new("increment", |mut state: TestState| {
                Box::pin(async move {
                    state.counter += 1;
                    Ok(state)
                })
            }),
        )
        .set_entry_point("increment")
        .compile()
        .unwrap();

    let config = ExecutionConfig::new("test-thread-2")
        .with_interrupts(interrupt_manager.clone())
        .add_interrupt_node("increment", InterruptReason::ApprovalRequired);

    let initial_state = TestState {
        counter: 0,
        message: String::new(),
    };

    // Execute in background
    let graph_clone = graph.clone();
    let execute_handle = tokio::spawn(async move {
        graph_clone
            .execute_with_config(initial_state, config)
            .await
    });

    // Wait for interrupt
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

    // Reject the interrupt
    let interrupts = interrupt_manager.get_active_interrupts().await;
    assert_eq!(interrupts.len(), 1);

    let response = InterruptResponse::reject(&interrupts[0].id);
    interrupt_manager.respond(response).await.unwrap();

    // Execution should fail
    let result = execute_handle.await.unwrap();
    assert!(result.is_err());
}

#[tokio::test]
async fn test_time_travel_debugging() {
    let history_manager = Arc::new(ExecutionHistoryManager::<TestState>::new());

    // Build a graph with multiple steps
    let graph = StateGraph::new()
        .add_node(
            "step1",
            FunctionNode::new("step1", |mut state: TestState| {
                Box::pin(async move {
                    state.counter += 1;
                    state.message = format!("Step 1: {}", state.counter);
                    Ok(state)
                })
            }),
        )
        .add_node(
            "step2",
            FunctionNode::new("step2", |mut state: TestState| {
                Box::pin(async move {
                    state.counter *= 2;
                    state.message = format!("Step 2: {}", state.counter);
                    Ok(state)
                })
            }),
        )
        .add_node(
            "step3",
            FunctionNode::new("step3", |mut state: TestState| {
                Box::pin(async move {
                    state.counter += 10;
                    state.message = format!("Step 3: {}", state.counter);
                    Ok(state)
                })
            }),
        )
        .add_edge("step1", "step2")
        .add_edge("step2", "step3")
        .set_entry_point("step1")
        .set_finish_point("step3")
        .compile()
        .unwrap();

    let config = ExecutionConfig::new("test-thread-3")
        .with_time_travel(history_manager.clone());

    let initial_state = TestState {
        counter: 5,
        message: String::new(),
    };

    // Execute the graph
    let result = graph.execute_with_config(initial_state, config).await.unwrap();
    assert_eq!(result.counter, 22); // (5 + 1) * 2 + 10 = 22

    // Get execution history
    let histories = history_manager.get_histories("test-thread-3").await;
    assert_eq!(histories.len(), 1);

    let history = &histories[0];
    assert_eq!(history.step_count(), 3);
    assert!(history.success);
    assert_eq!(history.execution_path(), vec!["step1", "step2", "step3"]);

    // Verify state at each step
    assert_eq!(history.state_at_step(0).unwrap().counter, 5); // Initial
    assert_eq!(history.state_at_step(1).unwrap().counter, 6); // After step1
    assert_eq!(history.state_at_step(2).unwrap().counter, 12); // After step2
    assert_eq!(history.state_at_step(3).unwrap().counter, 22); // After step3

    // Create a debugger
    let mut debugger = history_manager
        .create_debugger(&history.id)
        .await
        .unwrap();

    // Test step navigation
    assert_eq!(debugger.current_state().counter, 5); // Initial

    debugger.step_forward().unwrap();
    assert_eq!(debugger.current_state().counter, 6); // After step1

    debugger.step_forward().unwrap();
    assert_eq!(debugger.current_state().counter, 12); // After step2

    debugger.step_backward().unwrap();
    assert_eq!(debugger.current_state().counter, 6); // Back to after step1

    debugger.jump_to_end();
    assert_eq!(debugger.current_state().counter, 22); // Final state

    debugger.jump_to_start();
    assert_eq!(debugger.current_state().counter, 5); // Initial state

    // Test diff
    let diff = debugger.diff(0, 3).unwrap();
    assert_eq!(diff.state_from.counter, 5);
    assert_eq!(diff.state_to.counter, 22);
}

#[tokio::test]
async fn test_combined_hitl_and_time_travel() {
    let interrupt_manager = Arc::new(InterruptManager::<TestState>::new());
    let history_manager = Arc::new(ExecutionHistoryManager::<TestState>::new());

    // Build a graph with both features
    let graph = StateGraph::new()
        .add_node(
            "step1",
            FunctionNode::new("step1", |mut state: TestState| {
                Box::pin(async move {
                    state.counter += 1;
                    state.message = format!("Step 1: {}", state.counter);
                    Ok(state)
                })
            }),
        )
        .add_node(
            "step2",
            FunctionNode::new("step2", |mut state: TestState| {
                Box::pin(async move {
                    state.counter *= 2;
                    state.message = format!("Step 2: {}", state.counter);
                    Ok(state)
                })
            }),
        )
        .add_edge("step1", "step2")
        .set_entry_point("step1")
        .set_finish_point("step2")
        .compile()
        .unwrap();

    let config = ExecutionConfig::new("test-thread-4")
        .with_interrupts(interrupt_manager.clone())
        .with_time_travel(history_manager.clone())
        .add_interrupt_node(
            "step2",
            InterruptReason::InputRequired {
                prompt: "Approve step2 execution?".to_string(),
            },
        );

    let initial_state = TestState {
        counter: 10,
        message: String::new(),
    };

    // Execute in background
    let graph_clone = graph.clone();
    let execute_handle = tokio::spawn(async move {
        graph_clone
            .execute_with_config(initial_state, config)
            .await
    });

    // Wait for interrupt at step2
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // Check that step1 was recorded in history
    let active_history = history_manager
        .get_active_history("test-thread-4")
        .await
        .unwrap();
    assert_eq!(active_history.step_count(), 1); // Only step1 completed before interrupt

    // Approve the interrupt
    let interrupts = interrupt_manager.get_active_interrupts().await;
    assert_eq!(interrupts.len(), 1);
    assert_eq!(interrupts[0].node_name, "step2");
    assert_eq!(interrupts[0].state.counter, 11); // After step1

    let response = InterruptResponse::approve(&interrupts[0].id);
    interrupt_manager.respond(response).await.unwrap();

    // Wait for execution to complete
    let result = execute_handle.await.unwrap().unwrap();
    assert_eq!(result.counter, 22); // (10 + 1) * 2

    // Verify complete history
    let histories = history_manager.get_histories("test-thread-4").await;
    assert_eq!(histories.len(), 1);
    assert_eq!(histories[0].step_count(), 2);
    assert!(histories[0].success);
}

#[tokio::test]
async fn test_time_travel_with_error() {
    let history_manager = Arc::new(ExecutionHistoryManager::<TestState>::new());

    // Build a graph that will fail
    let graph = StateGraph::new()
        .add_node(
            "step1",
            FunctionNode::new("step1", |mut state: TestState| {
                Box::pin(async move {
                    state.counter += 1;
                    Ok(state)
                })
            }),
        )
        .add_node(
            "failing_step",
            FunctionNode::new("failing_step", |_state: TestState| {
                Box::pin(async move {
                    Err(Error::node_execution_failed("Intentional failure"))
                })
            }),
        )
        .add_edge("step1", "failing_step")
        .set_entry_point("step1")
        .compile()
        .unwrap();

    let config = ExecutionConfig::new("test-thread-5")
        .with_time_travel(history_manager.clone());

    let initial_state = TestState {
        counter: 0,
        message: String::new(),
    };

    // Execute (will fail)
    let result = graph.execute_with_config(initial_state, config).await;
    assert!(result.is_err());

    // Check history recorded the error
    let histories = history_manager.get_histories("test-thread-5").await;
    assert_eq!(histories.len(), 1);
    assert!(!histories[0].success);
    assert_eq!(histories[0].step_count(), 2); // step1 succeeded, failing_step failed
    assert!(histories[0].has_errors());

    let error_steps = histories[0].error_steps();
    assert_eq!(error_steps.len(), 1);
    assert_eq!(error_steps[0].node_name, "failing_step");
    assert!(error_steps[0].error.is_some());
}

#[tokio::test]
async fn test_interrupt_input_required() {
    let interrupt_manager = Arc::new(InterruptManager::<TestState>::new());

    let graph = StateGraph::new()
        .add_node(
            "process",
            FunctionNode::new("process", |state: TestState| {
                Box::pin(async move { Ok(state) })
            }),
        )
        .set_entry_point("process")
        .compile()
        .unwrap();

    let config = ExecutionConfig::new("test-thread-6")
        .with_interrupts(interrupt_manager.clone())
        .add_interrupt_node(
            "process",
            InterruptReason::InputRequired {
                prompt: "Enter your name:".to_string(),
            },
        );

    let initial_state = TestState {
        counter: 0,
        message: String::new(),
    };

    // Execute in background
    let graph_clone = graph.clone();
    let execute_handle = tokio::spawn(async move {
        graph_clone
            .execute_with_config(initial_state, config)
            .await
    });

    // Wait for interrupt
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

    // Check interrupt reason
    let interrupts = interrupt_manager.get_active_interrupts().await;
    assert_eq!(interrupts.len(), 1);

    match &interrupts[0].reason {
        InterruptReason::InputRequired { prompt } => {
            assert_eq!(prompt, "Enter your name:");
        }
        _ => panic!("Expected InputRequired interrupt"),
    }

    // Provide input
    let response = InterruptResponse::with_data(
        &interrupts[0].id,
        serde_json::json!({"name": "Alice"}),
    );
    interrupt_manager.respond(response).await.unwrap();

    // Wait for completion
    let result = execute_handle.await.unwrap();
    assert!(result.is_ok());
}
