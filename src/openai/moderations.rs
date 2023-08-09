use serde::{Deserialize, Serialize};

/// Main structure for the moderation response.
#[derive(Debug, Serialize, Deserialize)]
pub struct Response {
    /// Unique ID of the moderation request.
    pub id: String,

    /// Model identifier used for this moderation.
    pub model: String,

    /// List of results from the moderation.
    pub results: Vec<Result>,
}

/// Represents a single result from the moderation process.
#[derive(Debug, Serialize, Deserialize)]
pub struct Result {
    /// Flag indicating if the content was flagged by any category.
    pub flagged: bool,

    /// Details about the categories that were flagged.
    pub categories: Categories,

    /// Scores associated with each category.
    pub category_scores: Scores,
}

/// Categories that content can be flagged under.
#[derive(Debug, Serialize, Deserialize)]
pub struct Categories {
    /// Flag for sexual content.
    pub sexual: bool,

    /// Flag for hate speech.
    pub hate: bool,

    /// Flag for harassment.
    pub harassment: bool,

    /// Flag for self-harm content.
    #[serde(rename = "self-harm")]
    pub self_harm: bool,

    /// Flag for sexual content involving minors.
    #[serde(rename = "sexual/minors")]
    pub sexual_minors: bool,

    /// Flag for threatening hate speech.
    #[serde(rename = "hate/threatening")]
    pub hate_threatening: bool,

    /// Flag for graphic violence.
    #[serde(rename = "violence/graphic")]
    pub violence_graphic: bool,

    /// Flag for intent of self-harm.
    #[serde(rename = "self-harm/intent")]
    pub self_harm_intent: bool,

    /// Flag for instructions on self-harm.
    #[serde(rename = "self-harm/instructions")]
    pub self_harm_instructions: bool,

    /// Flag for threatening harassment.
    #[serde(rename = "harassment/threatening")]
    pub harassment_threatening: bool,

    /// Flag for violent content.
    pub violence: bool,
}

/// Scores associated with each moderation category.
#[derive(Debug, Serialize, Deserialize)]
pub struct Scores {
    /// Score for sexual content.
    pub sexual: f64,

    /// Score for hate speech.
    pub hate: f64,

    /// Score for harassment.
    pub harassment: f64,

    /// Score for self-harm content.
    #[serde(rename = "self-harm")]
    pub self_harm: f64,

    /// Score for sexual content involving minors.
    #[serde(rename = "sexual/minors")]
    pub sexual_minors: f64,

    /// Score for threatening hate speech.
    #[serde(rename = "hate/threatening")]
    pub hate_threatening: f64,

    /// Score for graphic violence.
    #[serde(rename = "violence/graphic")]
    pub violence_graphic: f64,

    /// Score for intent of self-harm.
    #[serde(rename = "self-harm/intent")]
    pub self_harm_intent: f64,

    /// Score for instructions on self-harm.
    #[serde(rename = "self-harm/instructions")]
    pub self_harm_instructions: f64,

    /// Score for threatening harassment.
    #[serde(rename = "harassment/threatening")]
    pub harassment_threatening: f64,

    /// Score for violent content.
    pub violence: f64,
}

/// Represents a `Moderation` object in the `OpenAI` moderation API.
///
/// For more information check the official [openAI API documentation](https://platform.openai.com/docs/api-reference/moderations)
#[derive(Debug, Serialize, Deserialize)]
pub struct Moderation {
    /// The input text to classify
    pub input: String,
}
