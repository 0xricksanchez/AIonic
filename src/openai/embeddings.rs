use crate::openai::misc::Usage;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum InputType {
    SingleString(String),
    MultipleStrings(Vec<String>),
    MultipleTokens(Vec<u64>),
}

impl InputType {
    pub fn is_single_string(&self) -> bool {
        matches!(self, InputType::SingleString(_))
    }

    pub fn is_multiple_strings(&self) -> bool {
        matches!(self, InputType::MultipleStrings(_))
    }

    pub fn is_multiple_tokens(&self) -> bool {
        matches!(self, InputType::MultipleTokens(_))
    }

    pub fn new_single_string(input: String) -> Self {
        InputType::SingleString(input)
    }

    pub fn new_multiple_strings(input: Vec<String>) -> Self {
        InputType::MultipleStrings(input)
    }

    pub fn new_multiple_tokens(input: Vec<u64>) -> Self {
        InputType::MultipleTokens(input)
    }
}

impl From<String> for InputType {
    fn from(input: String) -> Self {
        InputType::SingleString(input)
    }
}

impl From<&[u64]> for InputType {
    fn from(input: &[u64]) -> Self {
        InputType::MultipleTokens(input.to_vec())
    }
}

impl From<Vec<String>> for InputType {
    fn from(input: Vec<String>) -> Self {
        InputType::MultipleStrings(input)
    }
}

impl From<Vec<u64>> for InputType {
    fn from(input: Vec<u64>) -> Self {
        InputType::MultipleTokens(input)
    }
}

impl From<&str> for InputType {
    fn from(input: &str) -> Self {
        InputType::SingleString(input.to_string())
    }
}

#[derive(Deserialize, Debug, Clone)]
pub struct Response {
    pub object: String,
    pub data: Vec<Data>,
    pub model: String,
    pub usage: Usage,
}

#[derive(Deserialize, Debug, Clone)]
pub struct Data {
    pub object: String,
    pub embedding: Vec<f64>,
    pub index: u64,
}

/// OpenAIs embeddings that can be used to measure the relatedness of text strings.
/// Embeddings are commonly used for:
///  
///  * Search (where results are ranked by relevance to a query string)
///  * Clustering (where text strings are grouped by similarity)
///  * Recommendations (where items with related text strings are recommended)
///  * Anomaly detection (where outliers with little relatedness are identified)
///  * Diversity measurement (where similarity distributions are analyzed)
///  * Classification (where text strings are classified by their most similar label)
///
/// For more information check the official [openAI API documentation](https://platform.openai.com/docs/api-reference/embeddings)
///
/// # Example
///
/// ```rust
/// use aionic::openai::Embedding;
/// use aionic::openai::OpenAIConfig;
///
/// let chat = Embedding::default();
/// ```
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Embedding {
    /// ID of the model to use.
    pub model: String,

    /// Input text to embed, encoded as a string or array of tokens. To embed multiple inputs in a single request,
    /// pass an array of strings or array of token arrays.
    pub input: InputType,

    /// A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

impl Embedding {
    const DEFAULT_MODEL: &'static str = "text-embedding-ada-002";

    /// Returns the default model to be used by this AI system.
    ///
    /// # Returns
    ///
    /// This function returns a static string slice (`&'static str`) which represents the identifier of the default model used by the AI system.
    pub fn get_default_model() -> &'static str {
        Self::DEFAULT_MODEL
    }
}
