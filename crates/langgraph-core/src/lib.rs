//! # LangGraph Core
//!
//! Core graph engine for LangGraph with StateGraph and MessageGraph implementations.
//! Provides enterprise-grade, production-ready graph execution with async support.
//!
//! ## Features
//!
//! - **Enhanced Streaming**: Stream execution with events, checkpointing, and configuration
//! - **Human-in-the-Loop**: Pause/resume execution with approval gates and state modification
//! - **Time Travel Debugging**: Replay from checkpoints, state diffing, and execution timeline

pub mod error;
pub mod graph;
pub mod interrupt;
pub mod message;
pub mod metadata;
pub mod node;
pub mod reducer;
pub mod replay;
pub mod schema;
pub mod state;
pub mod streaming;

pub use error::{Error, Result};
pub use graph::{CompiledGraph, ExecutionConfig, StateGraph};
pub use interrupt::{
    ApprovalConfig, InterruptConfig, InterruptEvent, InterruptHandler, InterruptResponse,
    InterruptSignal, ModificationOperation, StateModification,
};
pub use message::{Message, MessageGraph};
pub use metadata::{
    ConditionalDecision, ExecutionTimeline, ExecutionTrace, NodeExecutionMetadata, TimelineEntry,
    TimelineEventType,
};
pub use node::Node;
pub use reducer::Reducer;
pub use replay::{
    CheckpointComparison, CheckpointData, CheckpointLoader, CheckpointMetadata, FieldChange,
    ReplayConfig, StateDiff, StateDiffer, TimeTravelDebugger, TimelinePoint,
};
pub use schema::StateSchema;
pub use state::State;
pub use streaming::{StreamConfig, StreamEvent};

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::{
        ApprovalConfig, CheckpointComparison, CheckpointData, CheckpointLoader,
        CheckpointMetadata, CompiledGraph, ConditionalDecision, Error, ExecutionConfig,
        ExecutionTimeline, ExecutionTrace, FieldChange, InterruptConfig, InterruptEvent,
        InterruptHandler, InterruptResponse, InterruptSignal, Message, MessageGraph,
        ModificationOperation, Node, NodeExecutionMetadata, Reducer, ReplayConfig, Result, State,
        StateGraph, StateModification, StateSchema, StateDiff, StateDiffer, StreamConfig,
        StreamEvent, TimeTravelDebugger, TimelineEntry, TimelineEventType, TimelinePoint,
    };
}
