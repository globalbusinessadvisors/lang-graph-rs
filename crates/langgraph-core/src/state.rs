//! State trait and implementations

use serde::{de::DeserializeOwned, Serialize};
use std::fmt::Debug;

/// Core state trait for graph execution
///
/// State represents the data flowing through the graph.
/// It must be cloneable, serializable, and thread-safe.
pub trait State: Clone + Send + Sync + Debug + Serialize + DeserializeOwned + 'static {}

// Blanket implementation for any type that meets the requirements
impl<T> State for T where T: Clone + Send + Sync + Debug + Serialize + DeserializeOwned + 'static {}
