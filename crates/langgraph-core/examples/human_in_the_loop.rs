//! Human-in-the-loop example
//!
//! This example demonstrates interactive graph execution with human intervention:
//! - Setting breakpoints before/after specific nodes
//! - Inspecting state at pause points
//! - Modifying state before continuing
//! - Aborting or redirecting execution flow
//! - Tracking execution history

use langgraph_core::prelude::*;
use langgraph_core::node::FunctionNode;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WorkflowState {
    task: String,
    status: String,
    priority: i32,
    approvals: Vec<String>,
    value: i32,
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Human-in-the-Loop Example ===\n");

    // Build a multi-step approval workflow
    let graph = StateGraph::new()
        .add_node(
            "initialize",
            FunctionNode::new("initialize", |mut state: WorkflowState| {
                Box::pin(async move {
                    println!("[Initialize] Setting up task: {}", state.task);
                    state.status = "initialized".to_string();
                    state.value = 10;
                    Ok(state)
                })
            }),
        )
        .add_node(
            "process",
            FunctionNode::new("process", |mut state: WorkflowState| {
                Box::pin(async move {
                    println!("[Process] Processing task with priority: {}", state.priority);
                    state.value *= 2;
                    state.status = "processed".to_string();
                    Ok(state)
                })
            }),
        )
        .add_node(
            "review",
            FunctionNode::new("review", |mut state: WorkflowState| {
                Box::pin(async move {
                    println!("[Review] Task under review - current value: {}", state.value);
                    state.status = "under_review".to_string();
                    Ok(state)
                })
            }),
        )
        .add_node(
            "approve",
            FunctionNode::new("approve", |mut state: WorkflowState| {
                Box::pin(async move {
                    println!("[Approve] Approving task");
                    state.approvals.push("manager".to_string());
                    state.status = "approved".to_string();
                    Ok(state)
                })
            }),
        )
        .add_node(
            "finalize",
            FunctionNode::new("finalize", |mut state: WorkflowState| {
                Box::pin(async move {
                    println!("[Finalize] Finalizing with {} approvals", state.approvals.len());
                    state.status = "completed".to_string();
                    Ok(state)
                })
            }),
        )
        .add_edge("initialize", "process")
        .add_edge("process", "review")
        .add_edge("review", "approve")
        .add_edge("approve", "finalize")
        .set_entry_point("initialize")
        .set_finish_point("finalize");

    let compiled = graph.compile()?;

    // Example 1: Breakpoint before review node
    println!("\n--- Example 1: Breakpoint Before Review ---");
    run_with_breakpoint_before(&compiled).await?;

    // Example 2: Breakpoint after processing
    println!("\n--- Example 2: Breakpoint After Process ---");
    run_with_breakpoint_after(&compiled).await?;

    // Example 3: State modification during execution
    println!("\n--- Example 3: State Modification ---");
    run_with_state_modification(&compiled).await?;

    // Example 4: Multiple breakpoints with trace
    println!("\n--- Example 4: Multiple Breakpoints with Trace ---");
    run_with_trace(&compiled).await?;

    println!("\n=== All examples completed! ===");
    Ok(())
}

/// Example 1: Pause before the review node
async fn run_with_breakpoint_before(graph: &CompiledGraph<WorkflowState>) -> Result<()> {
    let initial_state = WorkflowState {
        task: "Deploy to production".to_string(),
        status: "pending".to_string(),
        priority: 5,
        approvals: vec![],
        value: 0,
    };

    // Configure to break before the "review" node
    let config = InteractiveConfig::with_breakpoints_before(vec!["review"]);

    let (mut handle, execution_task) = graph.execute_interactive(initial_state, config).await?;

    // Wait for the breakpoint
    if let Some((point, state)) = handle.wait_for_interaction().await {
        println!(
            "⏸️  Paused at '{}' (before execution)",
            point.node_name
        );
        println!("   Current state: status={}, value={}", state.status, state.value);
        println!("   Step: {}", point.step_count);

        // Inspect state and decide to continue
        println!("   Human decision: Looks good, continuing...");
        handle.continue_execution()?;
    }

    let final_state = execution_task.await.unwrap()?;
    println!("✅ Final state: status={}, value={}", final_state.status, final_state.value);

    Ok(())
}

/// Example 2: Pause after the process node
async fn run_with_breakpoint_after(graph: &CompiledGraph<WorkflowState>) -> Result<()> {
    let initial_state = WorkflowState {
        task: "Update database schema".to_string(),
        status: "pending".to_string(),
        priority: 8,
        approvals: vec![],
        value: 0,
    };

    let config = InteractiveConfig::with_breakpoints_after(vec!["process"]);

    let (mut handle, execution_task) = graph.execute_interactive(initial_state, config).await?;

    if let Some((point, state)) = handle.wait_for_interaction().await {
        println!(
            "⏸️  Paused at '{}' (after execution)",
            point.node_name
        );
        println!("   State after processing: status={}, value={}", state.status, state.value);

        // Verify the processing was correct
        if state.value == 20 {
            println!("   Human decision: Processing correct, continuing...");
            handle.continue_execution()?;
        } else {
            println!("   Human decision: Unexpected value, aborting!");
            handle.abort("Unexpected processing result")?;
        }
    }

    match execution_task.await.unwrap() {
        Ok(final_state) => {
            println!("✅ Final state: status={}", final_state.status);
        }
        Err(e) => {
            println!("❌ Execution error: {}", e);
        }
    }

    Ok(())
}

/// Example 3: Modify state during execution
async fn run_with_state_modification(graph: &CompiledGraph<WorkflowState>) -> Result<()> {
    let initial_state = WorkflowState {
        task: "Critical security patch".to_string(),
        status: "pending".to_string(),
        priority: 3,
        approvals: vec![],
        value: 0,
    };

    let config = InteractiveConfig::with_breakpoints_before(vec!["approve"]);

    let (mut handle, execution_task) = graph.execute_interactive(initial_state, config).await?;

    if let Some((point, mut state)) = handle.wait_for_interaction().await {
        println!(
            "⏸️  Paused at '{}', about to approve",
            point.node_name
        );
        println!("   Original priority: {}", state.priority);

        // Human decides to escalate priority
        println!("   Human decision: This is critical! Escalating priority to 10");
        state.priority = 10;
        state.approvals.push("security_team".to_string());

        // Continue with modified state
        handle.continue_with(state)?;
    }

    let final_state = execution_task.await.unwrap()?;
    println!(
        "✅ Final state: priority={}, approvals={:?}",
        final_state.priority, final_state.approvals
    );

    Ok(())
}

/// Example 4: Multiple breakpoints with execution trace
async fn run_with_trace(graph: &CompiledGraph<WorkflowState>) -> Result<()> {
    let initial_state = WorkflowState {
        task: "Rollout feature flag".to_string(),
        status: "pending".to_string(),
        priority: 5,
        approvals: vec![],
        value: 0,
    };

    // Set breakpoints at multiple nodes
    let mut breakpoint_nodes = HashSet::new();
    breakpoint_nodes.insert("process".to_string());
    breakpoint_nodes.insert("review".to_string());

    let config = InteractiveConfig {
        strategies: vec![InterruptStrategy::BeforeNodes(breakpoint_nodes)],
        max_steps: Some(100),
        auto_checkpoint: false,
        thread_id: None,
        debug: true,
    };

    let (mut handle, execution_task) = graph.execute_interactive(initial_state, config).await?;

    let mut breakpoint_count = 0;

    // Handle all breakpoints
    loop {
        match handle.wait_for_interaction().await {
            Some((point, state)) => {
                breakpoint_count += 1;
                println!(
                    "⏸️  Breakpoint #{} at '{}' (step {})",
                    breakpoint_count, point.node_name, point.step_count
                );
                println!("   State: status={}, value={}", state.status, state.value);

                // Show execution trace so far
                let trace = handle.trace().await;
                println!("   Trace: {:?}", trace.node_sequence());

                // Continue execution
                handle.continue_execution()?;
            }
            None => break,
        }
    }

    let final_state = execution_task.await.unwrap()?;
    println!("\n✅ Execution completed!");
    println!("   Final status: {}", final_state.status);
    println!("   Total breakpoints: {}", breakpoint_count);

    // Show final execution trace
    let trace = handle.trace().await;
    println!("\n📊 Execution Trace:");
    println!("   Total steps: {}", trace.total_steps);
    println!("   Node sequence: {:?}", trace.node_sequence());
    println!("   Duration: {:?}", trace.end_time.unwrap().duration_since(trace.start_time).unwrap());

    Ok(())
}
