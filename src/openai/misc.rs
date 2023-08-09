use serde::{Deserialize, Serialize};
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

/// Represents an error returned from the `OpenAI`' API.
///
/// This struct is used to deserialize the JSON object that the `OpenAI`' API
/// returns when a request fails. The API's error object has a fixed structure,
/// so this struct can directly map to it.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenAIError {
    /// Contains specific details about the error.
    pub error: ErrorDetails,
}

/// Contains detailed information about an error from the `OpenAI`' API.
///
/// The fields in this struct correspond to the properties in the error object
/// returned by the `OpenAI`' API. They provide detailed information about what
/// went wrong with a request, which can be useful for debugging and error handling.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ErrorDetails {
    /// A human-readable message providing more details about the error.
    pub message: String,

    /// The type of error returned. This can be used to categorize errors.
    pub r#type: String,

    /// (Optional) The specific parameter in the request that caused the error.
    pub param: Option<String>,

    /// (Optional) A code corresponding to the error. If provided, this can be
    /// used to handle specific error types programmatically.
    pub code: Option<String>,
}
