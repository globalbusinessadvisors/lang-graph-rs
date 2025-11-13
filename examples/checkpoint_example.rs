//! Checkpoint example
//!
//! This example demonstrates persistent checkpointing with SQLite.

use langgraph_checkpoint::{Checkpoint, Checkpointer, SqliteCheckpointer};
use langgraph_core::Result;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
struct AppState {
    user_id: String,
    progress: i32,
    data: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Checkpoint Example ===\n");

    // Create an in-memory SQLite checkpointer
    let checkpointer = SqliteCheckpointer::in_memory()?;

    // Save a checkpoint
    let state = AppState {
        user_id: "user123".to_string(),
        progress: 50,
        data: "Processing...".to_string(),
    };

    let checkpoint = Checkpoint::new("thread-1", state.clone());
    let checkpoint_id = checkpointer.save(checkpoint).await?;
    println!("Saved checkpoint: {}", checkpoint_id);

    // Save another checkpoint
    let state2 = AppState {
        user_id: "user123".to_string(),
        progress: 75,
        data: "Almost done...".to_string(),
    };

    let checkpoint2 = Checkpoint::new("thread-1", state2.clone());
    checkpointer.save(checkpoint2).await?;
    println!("Saved second checkpoint\n");

    // List all checkpoints
    let checkpoints = checkpointer.list("thread-1").await?;
    println!("Found {} checkpoints:", checkpoints.len());
    for meta in &checkpoints {
        println!("  - {}: {}", meta.id, meta.created_at);
    }
    println!();

    // Load latest checkpoint
    let latest: Option<Checkpoint<AppState>> = checkpointer.load_latest("thread-1").await?;
    if let Some(checkpoint) = latest {
        println!("Latest checkpoint:");
        println!("  Progress: {}", checkpoint.state.progress);
        println!("  Data: {}", checkpoint.state.data);
    }

    // Count checkpoints
    let count = checkpointer.count("thread-1").await?;
    println!("\nTotal checkpoints for thread-1: {}", count);

    Ok(())
}
