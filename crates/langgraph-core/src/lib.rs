//! # LangGraph Core
//!
//! Core graph engine for LangGraph with StateGraph and MessageGraph implementations.
//! Provides enterprise-grade, production-ready graph execution with async support.

pub mod error;
pub mod graph;
pub mod interrupt;
pub mod message;
pub mod node;
pub mod reducer;
pub mod schema;
pub mod state;
pub mod timetravel;

pub use error::{Error, Result};
pub use graph::{CompiledGraph, ExecutionConfig, StateGraph};
pub use interrupt::{
    Interrupt, InterruptConfig, InterruptManager, InterruptReason, InterruptResponse,
};
pub use message::{Message, MessageGraph};
pub use node::{FunctionNode, Node};
pub use reducer::Reducer;
pub use schema::StateSchema;
pub use state::State;
pub use timetravel::{
    ExecutionHistory, ExecutionHistoryManager, ExecutionStep, StateDiff, TimeTravelDebugger,
};

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::{
        CompiledGraph, Error, ExecutionConfig, ExecutionHistory, ExecutionHistoryManager,
        ExecutionStep, FunctionNode, Interrupt, InterruptConfig, InterruptManager,
        InterruptReason, InterruptResponse, Message, MessageGraph, Node, Reducer, Result, State,
        StateGraph, StateSchema, StateDiff, TimeTravelDebugger,
    };
}
