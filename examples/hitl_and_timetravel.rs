//! Example demonstrating Human-in-the-loop and Time travel debugging features
//!
//! This example shows how to:
//! 1. Create a graph with interrupt points for human approval
//! 2. Enable automatic execution history tracking
//! 3. Use the time travel debugger to step through execution
//!
//! Run with: cargo run --example hitl_and_timetravel

use langgraph_core::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct OrderState {
    order_id: String,
    amount: f64,
    status: String,
    approvals: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("🚀 LangGraph Human-in-the-loop & Time Travel Example\n");

    // Create managers for both features
    let interrupt_manager = Arc::new(InterruptManager::<OrderState>::new());
    let history_manager = Arc::new(ExecutionHistoryManager::<OrderState>::new());

    // Build an order processing graph
    let graph = StateGraph::new()
        .add_node(
            "validate_order",
            FunctionNode::new("validate_order", |mut state: OrderState| {
                Box::pin(async move {
                    println!("  📋 Validating order {}...", state.order_id);
                    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                    state.status = "validated".to_string();
                    Ok(state)
                })
            }),
        )
        .add_node(
            "check_inventory",
            FunctionNode::new("check_inventory", |mut state: OrderState| {
                Box::pin(async move {
                    println!("  📦 Checking inventory for order {}...", state.order_id);
                    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                    state.status = "inventory_checked".to_string();
                    Ok(state)
                })
            }),
        )
        .add_node(
            "calculate_price",
            FunctionNode::new("calculate_price", |mut state: OrderState| {
                Box::pin(async move {
                    println!("  💰 Calculating price for order {}...", state.order_id);
                    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                    state.amount *= 1.1; // Add 10% tax
                    state.status = "price_calculated".to_string();
                    Ok(state)
                })
            }),
        )
        .add_node(
            "approve_order",
            FunctionNode::new("approve_order", |mut state: OrderState| {
                Box::pin(async move {
                    println!("  ✅ Order {} approved!", state.order_id);
                    state.status = "approved".to_string();
                    state.approvals.push("manager".to_string());
                    Ok(state)
                })
            }),
        )
        .add_node(
            "process_payment",
            FunctionNode::new("process_payment", |mut state: OrderState| {
                Box::pin(async move {
                    println!("  💳 Processing payment for order {}...", state.order_id);
                    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                    state.status = "paid".to_string();
                    Ok(state)
                })
            }),
        )
        .add_edge("validate_order", "check_inventory")
        .add_edge("check_inventory", "calculate_price")
        .add_edge("calculate_price", "approve_order")
        .add_edge("approve_order", "process_payment")
        .set_entry_point("validate_order")
        .set_finish_point("process_payment")
        .compile()?;

    println!("✨ Graph compiled successfully!\n");

    // Configure execution with interrupts and time travel
    let config = ExecutionConfig::new("order-123")
        .with_interrupts(interrupt_manager.clone())
        .with_time_travel(history_manager.clone())
        // Require human approval before finalizing
        .add_interrupt_node(
            "approve_order",
            InterruptReason::DecisionRequired {
                options: vec!["approve".to_string(), "reject".to_string()],
            },
        )
        .with_debug(true);

    let initial_state = OrderState {
        order_id: "ORD-12345".to_string(),
        amount: 100.0,
        status: "pending".to_string(),
        approvals: vec![],
    };

    println!("🎬 Starting order processing...\n");

    // Execute in background
    let graph_clone = graph.clone();
    let execute_handle = tokio::spawn(async move {
        graph_clone
            .execute_with_config(initial_state, config)
            .await
    });

    // Wait for the interrupt
    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

    println!("\n⏸️  Execution paused for human approval...\n");

    // Check active interrupts
    let interrupts = interrupt_manager.get_active_interrupts().await;
    if let Some(interrupt) = interrupts.first() {
        println!("📨 Interrupt Details:");
        println!("   Node: {}", interrupt.node_name);
        println!("   Order: {}", interrupt.state.order_id);
        println!("   Amount: ${:.2}", interrupt.state.amount);
        println!("   Status: {}", interrupt.state.status);

        match &interrupt.reason {
            InterruptReason::DecisionRequired { options } => {
                println!("   Decision required: {:?}", options);
            }
            _ => {}
        }

        println!("\n✅ Approving order...\n");

        // Simulate human approval
        let response = InterruptResponse::approve(&interrupt.id);
        interrupt_manager.respond(response).await?;
    }

    // Wait for execution to complete
    let final_state = execute_handle.await.unwrap()?;

    println!("\n🎉 Order processing completed!\n");
    println!("Final state:");
    println!("  Order ID: {}", final_state.order_id);
    println!("  Amount: ${:.2}", final_state.amount);
    println!("  Status: {}", final_state.status);
    println!("  Approvals: {:?}", final_state.approvals);

    // Demonstrate time travel debugging
    println!("\n⏮️  Time Travel Debugging\n");

    let histories = history_manager.get_histories("order-123").await;
    if let Some(history) = histories.first() {
        println!("📜 Execution History:");
        println!("   Total steps: {}", history.step_count());
        println!("   Duration: {}μs", history.total_duration_micros());
        println!("   Success: {}", history.success);
        println!("   Path: {:?}\n", history.execution_path());

        // Create debugger
        let mut debugger = history_manager.create_debugger(&history.id).await?;

        println!("🔍 Stepping through execution:\n");

        // Step through each state
        for i in 0..=history.step_count() {
            let state = debugger.current_state();
            if let Some(step) = debugger.current_step() {
                println!(
                    "  Step {}: {} - Status: {}, Amount: ${:.2}",
                    i, step.node_name, state.status, state.amount
                );
            } else {
                println!("  Initial: Status: {}, Amount: ${:.2}", state.status, state.amount);
            }

            if i < history.step_count() {
                debugger.step_forward()?;
            }
        }

        // Show diff between start and end
        println!("\n📊 State Changes:");
        let diff = debugger.diff(0, history.step_count())?;
        println!("  Initial amount: ${:.2}", diff.state_from.amount);
        println!("  Final amount: ${:.2}", diff.state_to.amount);
        println!("  Change: ${:.2}", diff.state_to.amount - diff.state_from.amount);
        println!("  Initial status: {}", diff.state_from.status);
        println!("  Final status: {}", diff.state_to.status);

        // Demonstrate jumping in time
        println!("\n⏪ Time travel demonstration:");
        debugger.jump_to(2)?; // Jump to step 2
        println!("  Jumped to step 2: {}", debugger.current_state().status);

        debugger.jump_to_start();
        println!("  Back to start: {}", debugger.current_state().status);

        debugger.jump_to_end();
        println!("  Jump to end: {}", debugger.current_state().status);
    }

    println!("\n✅ Example completed successfully!");

    Ok(())
}
