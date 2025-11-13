//! # LangGraph Core
//!
//! Core graph engine for LangGraph with StateGraph and MessageGraph implementations.
//! Provides enterprise-grade, production-ready graph execution with async support.

pub mod error;
pub mod graph;
pub mod message;
pub mod node;
pub mod reducer;
pub mod schema;
pub mod state;

pub use error::{Error, Result};
pub use graph::{CompiledGraph, StateGraph};
pub use message::{Message, MessageGraph};
pub use node::Node;
pub use reducer::Reducer;
pub use schema::StateSchema;
pub use state::State;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::{
        CompiledGraph, Error, Message, MessageGraph, Node, Reducer, Result, State, StateGraph,
        StateSchema,
    };
}
