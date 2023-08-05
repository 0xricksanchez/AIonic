use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, Copy)]
pub enum MessageRole {
    User,
    Assistant,
    System,
    Function,
}

impl ToString for MessageRole {
    fn to_string(&self) -> String {
        match self {
            Self::User => "user".to_string(),
            Self::Assistant => "assistant".to_string(),
            Self::System => "system".to_string(),
            Self::Function => "function".to_string(),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Message {
    pub role: String,
    pub content: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<FunctionCall>,
}

impl Message {
    pub fn new<S: Into<String>>(role: &MessageRole, content: S) -> Self {
        Self {
            role: role.to_string(),
            content: content.into(),
            name: None,
            function_call: None,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Chat {
    pub model: String,
    pub messages: Vec<Message>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub functions: Option<Vec<Function>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<HashMap<String, f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

impl Chat {
    const DEFAULT_TEMPERATURE: f64 = 1.0;
    const DEFAULT_MAX_TOKENS: u64 = 2048;
    const DEFAULT_STREAM_RESPONSE: bool = true;
    const DEFAULT_MODEL: &str = "gpt-3.5-turbo";
    pub fn get_default_temperature() -> f64 {
        Self::DEFAULT_TEMPERATURE
    }

    pub fn get_default_max_tokens() -> u64 {
        Self::DEFAULT_MAX_TOKENS
    }

    pub fn get_default_stream() -> bool {
        Self::DEFAULT_STREAM_RESPONSE
    }

    pub fn get_default_model() -> &'static str {
        Self::DEFAULT_MODEL
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Function {
    pub name: String,
    pub description: Option<String>,
    // FIXME: this should be a JSON Schema https://platform.openai.com/docs/api-reference/chat/create#chat/create-parameters
    pub parameters: String,
}
