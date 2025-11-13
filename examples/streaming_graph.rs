//! Streaming graph example
//!
//! This example demonstrates streaming execution with intermediate state updates.

use futures::{pin_mut, StreamExt};
use langgraph_core::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
struct StreamState {
    value: i32,
    operations: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Streaming Graph Example ===\n");

    // Create a graph with multiple nodes
    let graph = StateGraph::new()
        .add_node(
            "add",
            FunctionNode::new("add", |mut state: StreamState| {
                Box::pin(async move {
                    state.value += 10;
                    state.operations.push(format!("+10 = {}", state.value));
                    Ok(state)
                })
            }),
        )
        .add_node(
            "multiply",
            FunctionNode::new("multiply", |mut state: StreamState| {
                Box::pin(async move {
                    state.value *= 2;
                    state.operations.push(format!("*2 = {}", state.value));
                    Ok(state)
                })
            }),
        )
        .add_node(
            "subtract",
            FunctionNode::new("subtract", |mut state: StreamState| {
                Box::pin(async move {
                    state.value -= 5;
                    state.operations.push(format!("-5 = {}", state.value));
                    Ok(state)
                })
            }),
        )
        .add_edge("add", "multiply")
        .add_edge("multiply", "subtract")
        .set_entry_point("add")
        .set_finish_point("subtract")
        .compile()?;

    // Stream execution
    let stream = graph
        .stream(StreamState {
            value: 0,
            operations: Vec::new(),
        })
        .await?;

    pin_mut!(stream);

    println!("Streaming execution:\n");
    while let Some(result) = stream.next().await {
        match result {
            Ok((node_name, state)) => {
                println!("Node '{}' completed:", node_name);
                println!("  Current value: {}", state.value);
                println!("  Operations: {:?}\n", state.operations);
            }
            Err(e) => {
                eprintln!("Error: {}", e);
                break;
            }
        }
    }

    Ok(())
}
