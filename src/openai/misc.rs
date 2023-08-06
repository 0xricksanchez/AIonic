use serde::Deserialize;
// =-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
// COMMON STRUCT DEFINITIONS
// =-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

/// Represents the response from an API call to `OpenAI` when
/// checking a specific model by name
#[derive(Debug, Deserialize, Clone)]
pub struct Model {
    /// The ID of the model represented as a name
    pub id: String,

    /// Will default to "model".
    pub object: String,

    /// The owner of the fetched model
    pub owned_by: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ModelsResponse {
    pub data: Vec<Model>,
    pub object: String,
}

/// Represents the usage data from an API call.
///
/// This includes the number of tokens used for the prompt, the completion, and the total tokens.
#[derive(Deserialize, Debug, Clone)]
pub struct Usage {
    /// Number of tokens used in the prompt.
    pub prompt_tokens: u64,

    /// Number of tokens used in the completion.
    /// This will be present in chat completions but absent in embeddings.
    pub completion_tokens: Option<u64>,

    /// Total number of tokens used in the API call.
    pub total_tokens: u64,
}
