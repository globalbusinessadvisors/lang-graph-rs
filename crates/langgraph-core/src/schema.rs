//! State schema definition and validation

use crate::{Error, Result};
use serde_json::Value;
use std::collections::HashMap;

/// State schema definition and validation trait
pub trait StateSchema: Send + Sync {
    /// Validate that a state conforms to this schema
    fn validate(&self, state: &Value) -> Result<()>;

    /// Get the schema definition as JSON
    fn schema(&self) -> Value;

    /// Get field types
    fn fields(&self) -> HashMap<String, String>;
}

/// JSON Schema-based state validator
#[derive(Debug, Clone)]
pub struct JsonSchema {
    schema: Value,
    fields: HashMap<String, String>,
}

impl JsonSchema {
    /// Create a new JSON schema validator
    pub fn new(schema: Value) -> Self {
        let fields = Self::extract_fields(&schema);
        Self { schema, fields }
    }

    /// Extract field types from schema
    fn extract_fields(schema: &Value) -> HashMap<String, String> {
        let mut fields = HashMap::new();
        if let Some(properties) = schema.get("properties").and_then(|p| p.as_object()) {
            for (key, value) in properties {
                if let Some(type_str) = value.get("type").and_then(|t| t.as_str()) {
                    fields.insert(key.clone(), type_str.to_string());
                }
            }
        }
        fields
    }

    /// Basic validation helper
    fn validate_value(&self, value: &Value, expected_type: &str) -> Result<()> {
        let matches = match expected_type {
            "string" => value.is_string(),
            "number" => value.is_number(),
            "integer" => value.is_i64() || value.is_u64(),
            "boolean" => value.is_boolean(),
            "array" => value.is_array(),
            "object" => value.is_object(),
            "null" => value.is_null(),
            _ => true, // Unknown types pass through
        };

        if matches {
            Ok(())
        } else {
            Err(Error::state_validation_failed(format!(
                "Expected type {}, got {:?}",
                expected_type, value
            )))
        }
    }
}

impl StateSchema for JsonSchema {
    fn validate(&self, state: &Value) -> Result<()> {
        if !state.is_object() {
            return Err(Error::state_validation_failed(
                "State must be an object".to_string(),
            ));
        }

        let state_obj = state.as_object().unwrap();

        // Check required fields
        if let Some(required) = self.schema.get("required").and_then(|r| r.as_array()) {
            for req in required {
                if let Some(field_name) = req.as_str() {
                    if !state_obj.contains_key(field_name) {
                        return Err(Error::state_validation_failed(format!(
                            "Required field '{}' missing",
                            field_name
                        )));
                    }
                }
            }
        }

        // Validate field types
        if let Some(properties) = self.schema.get("properties").and_then(|p| p.as_object()) {
            for (key, field_value) in state_obj {
                if let Some(field_schema) = properties.get(key) {
                    if let Some(expected_type) = field_schema.get("type").and_then(|t| t.as_str())
                    {
                        self.validate_value(field_value, expected_type)?;
                    }
                }
            }
        }

        Ok(())
    }

    fn schema(&self) -> Value {
        self.schema.clone()
    }

    fn fields(&self) -> HashMap<String, String> {
        self.fields.clone()
    }
}

/// No-op schema that accepts any state
#[derive(Debug, Clone, Default)]
pub struct AnySchema;

impl StateSchema for AnySchema {
    fn validate(&self, _state: &Value) -> Result<()> {
        Ok(())
    }

    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "additionalProperties": true
        })
    }

    fn fields(&self) -> HashMap<String, String> {
        HashMap::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_json_schema_validation() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name"]
        });

        let validator = JsonSchema::new(schema);

        // Valid state
        let valid_state = json!({"name": "Alice", "age": 30});
        assert!(validator.validate(&valid_state).is_ok());

        // Missing required field
        let invalid_state = json!({"age": 30});
        assert!(validator.validate(&invalid_state).is_err());

        // Wrong type
        let invalid_state = json!({"name": 123});
        assert!(validator.validate(&invalid_state).is_err());
    }

    #[test]
    fn test_any_schema() {
        let schema = AnySchema;
        let state = json!({"anything": "goes"});
        assert!(schema.validate(&state).is_ok());
    }
}
