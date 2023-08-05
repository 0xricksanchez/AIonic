use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Represents the response from a chat model API call to OpenAI.
///
/// Contains fields that provide information about the model used, the choices made by the model,
/// the unique ID for the API call, and usage data regarding the number of tokens processed.
#[derive(Deserialize, Debug)]
pub struct Response {
    /// Unique ID for the API call.
    pub id: Option<String>,

    /// Type of the API object. For a chat model, this should be 'chat.completion'.
    pub object: Option<String>,

    /// UNIX timestamp indicating when the chat model was created.
    pub created: Option<u64>,

    /// The model that was used for the chat session.
    pub model: Option<String>,

    /// Choices made by the chat model during the conversation.
    pub choices: Option<Vec<Choice>>,

    /// Information on the number of tokens processed in the request.
    pub usage: Option<Usage>,
}

/// Represents the usage data from an API call.
///
/// This includes the number of tokens used for the prompt, the completion, and the total tokens.
#[derive(Deserialize, Debug)]
pub struct Usage {
    /// Number of tokens used in the prompt.
    pub prompt_tokens: u64,

    /// Number of tokens used in the completion.
    pub completion_tokens: u64,

    /// Total number of tokens used in the API call.
    pub total_tokens: u64,
}

/// Represents a choice made by the model in a chat API call.
#[derive(Deserialize, Debug)]
pub struct Choice {
    /// The message that corresponds to the choice made.
    pub message: Message,

    /// Reason for finishing the generation.
    pub finish_reason: String,

    /// Index of the choice in the list of choices.
    pub index: u64,
}

/// Represents the response from a streaming chat model API call to OpenAI.
#[derive(Serialize, Deserialize, Debug)]
pub struct StreamedReponse {
    /// Unique ID for the API call.
    pub id: String,

    /// Type of the API object. For a chat model, this should be 'chat.completion'.
    pub object: String,

    /// UNIX timestamp indicating when the chat model was created.
    pub created: u64,

    /// The model that was used for the chat session.
    pub model: String,

    /// Choices made by the chat model during the conversation.
    pub choices: Vec<StreamedChoices>,
}

/// Represents a choice made by the model in a streaming chat API call.
#[derive(Serialize, Deserialize, Debug)]
pub struct StreamedChoices {
    /// Index of the choice in the list of choices.
    pub index: u64,

    /// Information about the change made by the model.
    pub delta: Delta,

    /// Reason for finishing the generation.
    pub finish_reason: Option<String>,
}

/// Represents a change made by the model in a streaming chat API call.
#[derive(Serialize, Deserialize, Debug)]
pub struct Delta {
    /// Role of the author making the change.
    pub role: Option<String>,

    /// Content of the change made.
    pub content: Option<String>,
}

/// Enumeration of roles for authors of messages in a chat API call.
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

impl<T: Into<String>> From<T> for MessageRole {
    fn from(s: T) -> Self {
        match s.into().as_str() {
            "assistant" => Self::Assistant,
            "system" => Self::System,
            "function" => Self::Function,
            _ => Self::User,
        }
    }
}

/// Represents a single Message exchanged with the OpenAI API during a conversational model session.
///
/// `Message` struct is used to encapsulate the details of an individual message in the conversation. This includes the role of the author,
/// the content of the message, the name of the author if the role is 'function', and information about any function that should be called.
///
/// Each message sent or received in a conversational model session with OpenAI API will be represented by an instance of this struct.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Message {
    /// The role of the messages author. One of system, user, assistant, or function.
    pub role: String,

    /// The contents of the message. content is required for all messages, and may be null for
    /// assistant messages with function calls.
    pub content: String,

    /// The name of the author of this message. name is required if role is function, and it should
    /// be the name of the function whose response is in the content. May contain a-z, A-Z, 0-9,
    /// and underscores, with a maximum length of 64 characters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// The name and arguments of a function that should be called, as generated by the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<FunctionCall>,
}

impl Message {
    /// Constructs a new `Message` instance.
    ///
    /// This function is responsible for creating a new message object that will be sent to or received from the OpenAI API.
    ///
    /// # Arguments
    ///
    /// * `role`: The role that corresponds to the author of the message. It should be either "user", "assistant", or "system".
    /// * `content`: The text content of the message. This should be the input provided by the user or the generated response.
    ///
    /// # Examples
    ///
    /// ```
    /// use aionic::openai::chat::{MessageRole, Message};
    ///
    /// let user_message = Message::new(&MessageRole::User, "Hello, assistant!");
    /// ```
    pub fn new<S: Into<String>>(role: &MessageRole, content: S) -> Self {
        Self {
            role: role.to_string(),
            content: content.into(),
            name: None,
            function_call: None,
        }
    }
}

impl<T: Into<String>> From<T> for Message {
    fn from(s: T) -> Self {
        Self {
            role: MessageRole::User.to_string(),
            content: s.into(),
            name: None,
            function_call: None,
        }
    }
}

impl std::fmt::Display for Message {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let role = match self.role.as_str() {
            "assistant" => MessageRole::Assistant,
            "system" => MessageRole::System,
            "function" => MessageRole::Function,
            _ => MessageRole::User,
        };
        write!(f, "{}: {}", role.to_string(), self.content)
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct FunctionCall {
    /// The name of the function to call.
    pub name: String,

    /// The arguments to call the function with, as generated by the model in JSON format.
    ///
    /// Note that the model does not always generate valid JSON, and may hallucinate parameters
    /// not defined by your function schema. Validate the arguments in your code before calling
    /// your function.
    pub arguments: String,
}

/// This struct is used for chat completions with OpenAI's models.
/// It contains all the parameters that can be set for an API request.
///
/// All fields with an `Option` type can be omitted from the JSON payload,
/// thanks to the `skip_serializing_if` attribute.
///
/// For more information check the official [openAI API documentation](https://platform.openai.com/docs/api-reference/completions/create)
///
/// # Example
///
/// ```
/// use aionic::openai::Chat;
/// use aionic::openai::OpenAIConfig;
///
/// let chat = Chat::default();
///
/// ```
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Chat {
    /// ID of the model to use. You can use the List models API to see all of your available models
    pub model: String,

    /// A list of messages comprising the conversation so far
    pub messages: Vec<Message>,

    /// A list of functions the model may generate JSON inputs for.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub functions: Option<Vec<Function>>,

    /// Controls how the model responds to function calls. "none" means the model does not call a function,
    /// and responds to the end-user. "auto" means the model can pick between an end-user or calling a function.
    /// Specifying a particular function via {"name":\ "my_function"} forces the model to call that function.
    /// "none" is the default when no functions are present. "auto" is the default if functions are present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<String>,

    /// What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random,
    /// while lower values like 0.2 will make it more focused and deterministic.
    /// It's generally recommended to either alter this or `top_p` but not both.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,

    /// An alternative to sampling with temperature, called nucleus sampling, where the model considers the results
    /// of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability
    /// mass are considered.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,

    /// How many chat completion choices to generate for each input message.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<i64>,

    /// If set, partial message deltas will be sent, like in ChatGPT.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    /// Up to 4 sequences where the API will stop generating further tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<String>,

    /// The maximum number of tokens to generate in the chat completion.
    /// The total length of input tokens and generated tokens is limited by the model's context length.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,

    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text
    /// so far, increasing the model's likelihood to talk about new topics.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,

    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in
    /// the text so far, decreasing the model's likelihood to repeat the same line verbatim.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,

    /// Modify the likelihood of specified tokens appearing in the completion.
    ///
    /// Accepts a json object that maps tokens (specified by their token ID in the tokenizer)
    /// to an associated bias value from -100 to 100. Mathematically, the bias is added to the
    /// logits generated by the model prior to sampling. The exact effect will vary per model,
    /// but values between -1 and 1 should decrease or increase likelihood of selection; values
    /// like -100 or 100 should result in a ban or exclusive selection of the relevant token.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<HashMap<String, f32>>,

    /// A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

impl Chat {
    const DEFAULT_TEMPERATURE: f64 = 1.0;
    const DEFAULT_MAX_TOKENS: u64 = 2048;
    const DEFAULT_STREAM_RESPONSE: bool = true;
    const DEFAULT_MODEL: &str = "gpt-3.5-turbo";
    /// Returns the default temperature for this AI system.
    ///
    /// # Returns
    ///
    /// This function returns a `f64` value which represents the default temperature.
    pub fn get_default_temperature() -> f64 {
        Self::DEFAULT_TEMPERATURE
    }

    /// Returns the default maximum token limit for this AI system.
    ///
    /// # Returns
    ///
    /// This function returns a `u64` value which represents the default maximum number of tokens that can be used in a single AI system action.
    pub fn get_default_max_tokens() -> u64 {
        Self::DEFAULT_MAX_TOKENS
    }

    /// Returns the default streaming behavior for this AI system.
    ///
    /// # Returns
    ///
    /// This function returns a `bool` value which represents the default behavior of the AI system when handling streaming data.
    /// If it returns `true`, the system will stream the data by default. If `false`, it will not.
    pub fn get_default_stream() -> bool {
        Self::DEFAULT_STREAM_RESPONSE
    }

    /// Returns the default model to be used by this AI system.
    ///
    /// # Returns
    ///
    /// This function returns a static string slice (`&'static str`) which represents the identifier of the default model used by the AI system.
    pub fn get_default_model() -> &'static str {
        Self::DEFAULT_MODEL
    }
}

/// This struct is used to describe a single function the model may generate JSON inputs for.
/// It's part of the `Chat` structure.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Function {
    /// The name of the function to be called. Must be a-z, A-Z, 0-9, or contain underscores and dashes, with a maximum length of 64.
    pub name: String,

    /// A description of what the function does, used by the model to choose when and how to call the function.
    pub description: Option<String>,

    /// The parameters the functions accepts, described as a JSON Schema object. See the guide for examples, and the JSON Schema
    /// reference for documentation about the format.
    ///
    /// To describe a function that accepts no parameters, provide the value {"type": "object", "properties": {}}.
    // FIXME:
    pub parameters: String,
}
