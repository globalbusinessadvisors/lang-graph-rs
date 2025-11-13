//! Message types for MessageGraph

use crate::{CompiledGraph, Node, Result, StateGraph};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Message role in a conversation
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Function,
}

/// A message in a conversation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Message {
    pub role: Role,
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl Message {
    /// Create a new message
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            name: None,
            metadata: HashMap::new(),
        }
    }

    /// Create a system message
    pub fn system(content: impl Into<String>) -> Self {
        Self::new(Role::System, content)
    }

    /// Create a user message
    pub fn user(content: impl Into<String>) -> Self {
        Self::new(Role::User, content)
    }

    /// Create an assistant message
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(Role::Assistant, content)
    }

    /// Create a function message
    pub fn function(content: impl Into<String>) -> Self {
        Self::new(Role::Function, content)
    }

    /// Set the name of the message sender
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Add metadata to the message
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

/// State for message-based graphs
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MessageState {
    pub messages: Vec<Message>,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

impl MessageState {
    /// Create a new message state
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            extra: HashMap::new(),
        }
    }

    /// Create from messages
    pub fn from_messages(messages: Vec<Message>) -> Self {
        Self {
            messages,
            extra: HashMap::new(),
        }
    }

    /// Add a message
    pub fn add_message(&mut self, message: Message) {
        self.messages.push(message);
    }

    /// Get the last message
    pub fn last_message(&self) -> Option<&Message> {
        self.messages.last()
    }

    /// Get all messages with a specific role
    pub fn messages_by_role(&self, role: &Role) -> Vec<&Message> {
        self.messages.iter().filter(|m| &m.role == role).collect()
    }
}

impl Default for MessageState {
    fn default() -> Self {
        Self::new()
    }
}

/// MessageGraph builder for conversation-based workflows
pub struct MessageGraph {
    state_graph: StateGraph<MessageState>,
}

impl MessageGraph {
    /// Create a new MessageGraph
    pub fn new() -> Self {
        Self {
            state_graph: StateGraph::new(),
        }
    }

    /// Add a node to the graph
    pub fn add_node<N>(mut self, name: impl Into<String>, node: N) -> Self
    where
        N: Node<MessageState> + 'static,
    {
        self.state_graph = self.state_graph.add_node(name, node);
        self
    }

    /// Add an edge between nodes
    pub fn add_edge(mut self, from: impl Into<String>, to: impl Into<String>) -> Self {
        self.state_graph = self.state_graph.add_edge(from, to);
        self
    }

    /// Add a conditional edge
    pub fn add_conditional_edge<F>(
        mut self,
        from: impl Into<String>,
        condition: F,
        edge_map: HashMap<String, String>,
    ) -> Self
    where
        F: Fn(&MessageState) -> String + Send + Sync + 'static,
    {
        self.state_graph = self.state_graph.add_conditional_edge(from, condition, edge_map);
        self
    }

    /// Set the entry point
    pub fn set_entry_point(mut self, name: impl Into<String>) -> Self {
        self.state_graph = self.state_graph.set_entry_point(name);
        self
    }

    /// Set the finish point
    pub fn set_finish_point(mut self, name: impl Into<String>) -> Self {
        self.state_graph = self.state_graph.set_finish_point(name);
        self
    }

    /// Compile the graph
    pub fn compile(self) -> Result<CompiledGraph<MessageState>> {
        self.state_graph.compile()
    }
}

impl Default for MessageGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_creation() {
        let msg = Message::user("Hello").with_name("Alice");
        assert_eq!(msg.role, Role::User);
        assert_eq!(msg.content, "Hello");
        assert_eq!(msg.name, Some("Alice".to_string()));
    }

    #[test]
    fn test_message_state() {
        let mut state = MessageState::new();
        state.add_message(Message::user("Hello"));
        state.add_message(Message::assistant("Hi there!"));

        assert_eq!(state.messages.len(), 2);
        assert_eq!(state.last_message().unwrap().content, "Hi there!");
        assert_eq!(state.messages_by_role(&Role::User).len(), 1);
    }
}
