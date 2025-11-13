//! Simple graph example
//!
//! This example demonstrates a basic workflow with two nodes.

use langgraph_core::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
struct WorkflowState {
    step: usize,
    data: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Simple Graph Example ===\n");

    // Create a simple graph with two nodes
    let graph = StateGraph::new()
        .add_node(
            "process",
            FunctionNode::new("process", |mut state: WorkflowState| {
                Box::pin(async move {
                    state.step += 1;
                    state.data = format!("Processed at step {}", state.step);
                    println!("Node 'process': {}", state.data);
                    Ok(state)
                })
            }),
        )
        .add_node(
            "finalize",
            FunctionNode::new("finalize", |mut state: WorkflowState| {
                Box::pin(async move {
                    state.step += 1;
                    state.data = format!("{} | Finalized at step {}", state.data, state.step);
                    println!("Node 'finalize': {}", state.data);
                    Ok(state)
                })
            }),
        )
        .add_edge("process", "finalize")
        .set_entry_point("process")
        .set_finish_point("finalize")
        .compile()?;

    // Execute the graph
    let result = graph
        .execute(WorkflowState {
            step: 0,
            data: String::from("Initial"),
        })
        .await?;

    println!("\nFinal state:");
    println!("  Step: {}", result.step);
    println!("  Data: {}", result.data);

    Ok(())
}
