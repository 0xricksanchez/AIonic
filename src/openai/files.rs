use serde::{Deserialize, Serialize};

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

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Response {
    pub data: Vec<Data>,
    pub object: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Data {
    pub id: String,
    pub object: String,
    pub bytes: u64,
    pub created_at: u64,
    pub filename: String,
    pub purpose: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct DeleteResponse {
    pub id: String,
    pub object: String,
    pub deleted: bool,
}
