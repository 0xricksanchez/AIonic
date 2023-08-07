use serde::{Deserialize, Serialize};

/// Represents a file in the OpenAI Files API.
///
/// A file may be uploaded for various purposes, and it's represented by a unique ID.
///
/// For more information check the official [openAI API documentation](https://platform.openai.com/docs/api-reference/files)
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Files {
    /// Name of the JSON Lines file to be uploaded.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,

    //The intended purpose of the uploaded documents.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub purpose: Option<String>,

    //The ID of the file to use for this request
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_id: Option<String>,
}

/// Represents the response from the OpenAI Files API.
///
/// It includes metadata and a list of data objects, each representing a file.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Response {
    /// List of `Data` objects each representing a file.
    pub data: Vec<Data>,

    /// The type of the object returned by the API.
    pub object: String,
}

/// Represents a file data in the OpenAI Files API.
///
/// It includes the file's unique id, the type of the object,
/// the number of bytes, creation timestamp, the file's name and its purpose.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Data {
    /// Unique ID of the file.
    pub id: String,

    /// The type of the object.
    pub object: String,

    /// The size of the file in bytes.
    pub bytes: u64,

    /// The timestamp at which the file was created.
    pub created_at: u64,

    /// The name of the file.
    pub filename: String,

    /// The intended purpose of the file.
    pub purpose: String,
}

/// Represents the response from the OpenAI Files API when a file is deleted successfully.
///
/// It includes metadata about the deleted file.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DeleteResponse {
    /// The type of the object returned by the API.
    pub object: String,

    /// Unique ID of the file that was deleted.
    pub id: String,

    /// Flag indicating whether the file was deleted successfully.
    pub deleted: bool,
}

/// Represents a prompt-completion pair in a JSONL response from the OpenAI API.
///
/// This is used for responses from the retrieve_content endpoint.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PromptCompletion {
    /// The prompt that was sent to the OpenAI API.
    pub prompt: String,

    /// The completion that was received from the OpenAI API.
    pub completion: String,
}
