//! Conditional graph example
//!
//! This example demonstrates conditional edges and looping.

use langgraph_core::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CounterState {
    count: i32,
    history: Vec<i32>,
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Conditional Graph Example ===\n");

    // Define conditional edges
    let mut edge_map = HashMap::new();
    edge_map.insert("continue".to_string(), "increment".to_string());
    edge_map.insert("done".to_string(), "finish".to_string());

    // Create a graph with conditional looping
    let graph = StateGraph::new()
        .add_node(
            "increment",
            FunctionNode::new("increment", |mut state: CounterState| {
                Box::pin(async move {
                    state.count += 1;
                    state.history.push(state.count);
                    println!("Count: {}", state.count);
                    Ok(state)
                })
            }),
        )
        .add_node(
            "finish",
            FunctionNode::new("finish", |state: CounterState| {
                Box::pin(async move {
                    println!("\nFinished! History: {:?}", state.history);
                    Ok(state)
                })
            }),
        )
        .add_conditional_edge(
            "increment",
            |state: &CounterState| {
                if state.count < 5 {
                    "continue".to_string()
                } else {
                    "done".to_string()
                }
            },
            edge_map,
        )
        .set_entry_point("increment")
        .set_finish_point("finish")
        .compile()?;

    // Execute the graph
    let result = graph
        .execute(CounterState {
            count: 0,
            history: Vec::new(),
        })
        .await?;

    println!("\nFinal count: {}", result.count);
    println!("Total iterations: {}", result.history.len());

    Ok(())
}
