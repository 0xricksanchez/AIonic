use reqwest::multipart::{Form, Part};
use reqwest::{Body, Client, IntoUrl};
use tokio_util::codec::{BytesCodec, FramedRead};

use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::error::Error;
use std::io::{self, Write};
use std::path::Path;
use std::process::exit;

#[derive(Debug, Deserialize)]
pub struct Model {
    pub id: String,
    pub object: String,
    pub owned_by: String,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct ModelsResponse {
    data: Vec<Model>,
    object: String,
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

#[allow(dead_code)]
#[derive(Deserialize, Debug)]
struct Response {
    id: Option<String>,
    object: Option<String>,
    created: Option<u64>,
    model: Option<String>,
    choices: Option<Vec<Choice>>,
    usage: Option<Usage>,
}

#[allow(dead_code)]
#[derive(Deserialize, Debug)]
struct Usage {
    prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
}

#[allow(dead_code)]
#[derive(Deserialize, Debug)]
struct Choice {
    message: Message,
    finish_reason: String,
    index: u64,
}

#[derive(Serialize, Deserialize, Debug)]
struct StreamedReponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<StreamedChoices>,
}

#[derive(Serialize, Deserialize, Debug)]
struct StreamedChoices {
    index: u64,
    delta: Delta,
    finish_reason: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
struct Delta {
    role: Option<String>,
    content: Option<String>,
}

pub trait OpenAIConfig: Send + Sync {
    fn default() -> Self;
}

impl OpenAIConfig for Chat {
    fn default() -> Self {
        Self {
            model: Self::get_default_model().into(),
            messages: vec![],
            functions: None,
            temperature: Some(Self::get_default_temperature()),
            top_p: None,
            n: None,
            stream: Some(Self::get_default_stream()),
            stop: None,
            max_tokens: Some(Self::get_default_max_tokens()),
            presence_penalty: None,
            frequency_penalty: None,
            logit_bias: None,
            user: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct OpenAIClient<C: OpenAIConfig> {
    pub client: Client,
    pub api_key: String,
    pub disable_live_stream: bool,
    pub config: C,
}

impl<C: OpenAIConfig + Serialize + Sync + Send + std::fmt::Debug> Default for OpenAIClient<C> {
    fn default() -> Self {
        Self::new()
    }
}

pub enum ImageResponseFormat {
    Url,
    Base64Json,
}

impl ToString for ImageResponseFormat {
    fn to_string(&self) -> String {
        match self {
            Self::Url => "url".to_string(),
            Self::Base64Json => "b64_json".to_string(),
        }
    }
}

#[derive(Clone, Debug, Copy)]
pub struct ImageSize {
    width: u64,
    height: u64,
}

impl ImageSize {
    pub fn new(width: u64, height: u64) -> Self {
        Self { width, height }
    }

    pub fn resize(mut self, width: Option<u64>, height: Option<u64>) -> Self {
        if let Some(width) = width {
            self.width = width;
        }
        if let Some(height) = height {
            self.height = height;
        }
        self
    }
}

impl ToString for ImageSize {
    fn to_string(&self) -> String {
        format!("{}x{}", self.width, self.height)
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Image {
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    n: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    size: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    image: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mask: Option<String>,
}

impl Image {
    const DEFAULT_N: u64 = 1;
    const DEFAULT_SIZE: &str = "1024x1024";
    const DEFAULT_RESPONSE_FORMAT: &str = "url";
    pub fn get_default_n() -> u64 {
        Self::DEFAULT_N
    }

    pub fn get_default_size() -> &'static str {
        Self::DEFAULT_SIZE
    }

    pub fn get_default_response_format() -> &'static str {
        Self::DEFAULT_RESPONSE_FORMAT
    }
}

impl OpenAIClient<Image> {
    pub fn set_response_format(mut self, response_format: &ImageResponseFormat) -> Self {
        self.config.response_format = Some(response_format.to_string());
        self
    }

    pub fn set_max_images(mut self, number_of_images: u64) -> Self {
        self.config.n = Some(number_of_images);
        self
    }

    pub fn set_size(mut self, size: &ImageSize) -> Self {
        self.config.size = Some(size.to_string());
        self
    }
}

impl OpenAIConfig for Image {
    fn default() -> Self {
        Self {
            prompt: None,
            n: Some(Self::get_default_n()),
            size: Some(Self::get_default_size().into()),
            response_format: Some(Self::get_default_response_format().into()),
            user: None,
            image: None,
            mask: None,
        }
    }
}

#[derive(Deserialize, Debug)]
pub struct ImageResponse {
    pub created: u64,
    pub data: Vec<ImageData>,
}

#[derive(Deserialize, Debug)]
pub struct ImageData {
    pub url: Option<String>,
    pub b64_json: Option<String>,
}

impl OpenAIClient<Image> {
    const OPENAI_API_IMAGE_GEN_URL: &str = "https://api.openai.com/v1/images/generations";
    const OPENAI_API_IMAGE_EDIT_URL: &str = "https://api.openai.com/v1/images/edits";
    const OPENAI_API_IMAGE_VARIATION_URL: &str = "https://api.openai.com/v1/images/variations";

    pub async fn create_image<S: Into<String>>(
        &mut self,
        prompt: S,
    ) -> Result<Vec<String>, Box<dyn Error + Send + Sync>> {
        self.config.prompt = Some(prompt.into());
        if self.config.image.is_some() {
            self.config.image = None;
        }
        if self.config.mask.is_some() {
            self.config.mask = None;
        }
        let image_response: ImageResponse = self
            ._make_post_request(Self::OPENAI_API_IMAGE_GEN_URL)
            .await?
            .json()
            .await?;

        Ok(self._parse_response(&image_response))
    }

    pub async fn edit_image<S: Into<String>>(
        &mut self,
        prompt: S,
        image_file_path: S,
        mask: Option<S>,
    ) -> Result<Vec<String>, Box<dyn Error + Send + Sync>> {
        self.config.image = Some(image_file_path.into());
        if let Some(mask) = mask {
            self.config.mask = Some(mask.into());
        }
        self.config.prompt = Some(prompt.into());
        println!("{:?}", self.config);

        let image_response: ImageResponse = self
            ._make_file_upload_request(Self::OPENAI_API_IMAGE_EDIT_URL)
            .await?;
        println!("{:?}", image_response);

        Ok(self._parse_response(&image_response))
    }

    async fn _get_streamed_body<P: AsRef<Path>>(
        &mut self,
        path: P,
    ) -> Result<Body, Box<dyn Error + Send + Sync>> {
        if !path.as_ref().exists() {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Image not found",
            )));
        }
        let file_stream_body = tokio::fs::File::open(path).await?;
        let stream = FramedRead::new(file_stream_body, BytesCodec::new());
        let body = Body::wrap_stream(stream);
        Ok(body)
    }

    async fn _create_file_part<P: AsRef<Path>>(
        &mut self,
        path: P,
    ) -> Result<Part, Box<dyn Error + Send + Sync>> {
        let file_name = path.as_ref().to_str().unwrap().to_string();
        let streamed_body = self._get_streamed_body(path).await?;
        let part_stream = Part::stream(streamed_body)
            .file_name(file_name)
            .mime_str("application/octet-stream")?;
        Ok(part_stream)
    }

    pub async fn create_image_variation<S: Into<String>>(
        &mut self,
        image_file_path: S,
    ) -> Result<Vec<String>, Box<dyn Error + Send + Sync>> {
        self.config.image = Some(image_file_path.into());
        if self.config.prompt.is_some() {
            self.config.prompt = None;
        }
        if self.config.mask.is_some() {
            self.config.mask = None;
        }
        let image_response: ImageResponse = self
            ._make_file_upload_request(Self::OPENAI_API_IMAGE_VARIATION_URL)
            .await?;

        Ok(self._parse_response(&image_response))
    }

    fn _parse_response(&mut self, image_response: &ImageResponse) -> Vec<String> {
        image_response
            .data
            .iter()
            .filter_map(|d| {
                if self.config.response_format == Some("url".into()) {
                    d.url.clone()
                } else {
                    d.b64_json.clone()
                }
            })
            .collect::<Vec<String>>()
    }

    async fn _make_file_upload_request<S: IntoUrl + Send + Sync>(
        &mut self,
        url: S,
    ) -> Result<ImageResponse, Box<dyn Error + Send + Sync>> {
        let file_name = self.config.image.clone().unwrap();
        let file_part_stream = self._create_file_part(file_name).await?;
        let mut form = Form::new().part("image", file_part_stream);

        if self.config.prompt.is_some() {
            form = form.text("prompt", self.config.prompt.clone().unwrap());
        }
        if self.config.mask.is_some() {
            let mask_name = self.config.mask.clone().unwrap();
            let mask_part_stream = self._create_file_part(mask_name).await?;
            form = form.part("mask", mask_part_stream);
        }

        if self.config.response_format.is_some() {
            form = form.text(
                "response_format",
                self.config.response_format.clone().unwrap(),
            );
        }

        if self.config.size.is_some() {
            form = form.text("size", self.config.size.clone().unwrap());
        }

        if self.config.n.is_some() {
            form = form.text("n", self.config.n.unwrap().to_string());
        }

        if self.config.user.is_some() {
            form = form.text("user", self.config.user.clone().unwrap());
        }

        let image_response: ImageResponse = self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .multipart(form)
            .send()
            .await?
            .json()
            .await?;
        Ok(image_response)
    }
}

impl OpenAIClient<Chat> {
    const OPENAI_API_COMPLETIONS_URL: &str = "https://api.openai.com/v1/chat/completions";

    pub fn with_config(mut self, config: Chat) -> Self {
        self.config = config;
        self
    }

    pub fn set_model(mut self, model: String) -> Self {
        self.config.model = model;
        self
    }

    pub fn set_max_tokens(mut self, max_tokens: u64) -> Self {
        self.config.max_tokens = Some(max_tokens);
        self
    }

    pub fn set_messages(mut self, messages: Vec<Message>) -> Self {
        self.config.messages = messages;
        self
    }

    pub fn set_temperature(mut self, temperature: f64) -> Self {
        self.config.temperature = Some(temperature);
        self
    }

    pub fn set_stream_responses(mut self, streamed: bool) -> Self {
        self.config.stream = Some(streamed);
        self
    }

    pub fn set_primer<S: Into<String>>(mut self, primer_msg: S) -> Self {
        let msg = Message::new(&MessageRole::System, primer_msg.into());
        self.config.messages.insert(0, msg);
        self
    }

    pub fn get_last_message(&self) -> Option<&Message> {
        self.config.messages.last()
    }

    fn _process_delta(
        &self,
        line: &str,
        answer_text: &mut Vec<String>,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        line.strip_prefix("data: ").map_or(Ok(()), |chunk| {
            if chunk.starts_with("[DONE]") {
                return Ok(());
            }
            let serde_chunk: Result<StreamedReponse, _> = serde_json::from_str(chunk);
            match serde_chunk {
                Ok(chunk) => {
                    for choice in chunk.choices {
                        if let Some(content) = choice.delta.content {
                            let sanitized_content =
                                content.trim().strip_suffix('\n').unwrap_or(&content);
                            if !self.disable_live_stream {
                                print!("{}", sanitized_content);
                                io::stdout().flush()?;
                            }
                            answer_text.push(sanitized_content.to_string());
                        }
                    }
                    Ok(())
                }
                Err(_) => Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "Deserialization Error",
                ))),
            }
        })
    }

    async fn _ask_openai_streamed(
        &mut self,
        res: &mut reqwest::Response,
        answer_text: &mut Vec<String>,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        print!("AI: ");
        loop {
            let chunk = match res.chunk().await {
                Ok(Some(chunk)) => chunk,
                Ok(None) => break,
                Err(e) => return Err(Box::new(e)),
            };
            let chunk_str = String::from_utf8_lossy(&chunk);
            let lines: Vec<&str> = chunk_str.split('\n').collect();
            for line in lines {
                self._process_delta(line, answer_text)?;
            }
        }
        println!();
        Ok(())
    }

    pub async fn ask<S: Into<String>>(
        &mut self,
        prompt: S,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let mut answer_chunks: Vec<String> = Vec::new();
        let is_streamed = self.config.stream.unwrap_or(false);
        self.config
            .messages
            .push(Message::new(&MessageRole::User, prompt));
        let r = self._make_post_request(Self::OPENAI_API_COMPLETIONS_URL);
        if is_streamed {
            let mut res = r.await?;
            self._ask_openai_streamed(&mut res, &mut answer_chunks)
                .await?;
        } else {
            let r = r.await?.json::<Response>().await?;
            for choice in r.choices.unwrap() {
                if !self.disable_live_stream {
                    print!("AI: {}\n", choice.message.content);
                    io::stdout().flush()?;
                }
                answer_chunks.push(choice.message.content);
            }
        }

        let answer_text = answer_chunks.join("");
        self.config
            .messages
            .push(Message::new(&MessageRole::Assistant, &answer_text));
        Ok(answer_text)
    }

    pub async fn chat(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut rl = DefaultEditor::new()?;
        let prompt = ">>> ";
        loop {
            let readline = rl.readline(prompt);
            match readline {
                Ok(line) => {
                    self.ask(line).await?;
                    println!();
                }
                Err(ReadlineError::Interrupted) => {
                    println!("CTRL-C");
                    break;
                }
                Err(ReadlineError::Eof) => {
                    println!("CTRL-D");
                    break;
                }
                Err(err) => {
                    println!("Error: {:?}", err);
                    break;
                }
            }
        }
        println!("{:?}", self.config.messages);
        Ok(())
    }
}

impl<C: OpenAIConfig + Serialize + std::fmt::Debug> OpenAIClient<C> {
    const OPENAI_API_MODELS_URL: &str = "https://api.openai.com/v1/models";
    pub fn new() -> Self {
        env::var("OPENAI_API_KEY").map_or_else(
            |_| {
                println!("OPENAI_API_KEY environment variable not set");
                exit(1);
            },
            |api_key| {
                let client = Client::new();
                Self {
                    client,
                    api_key,
                    disable_live_stream: false,
                    config: C::default(),
                }
            },
        )
    }

    pub fn disable_stdout(mut self) -> Self {
        self.disable_live_stream = true;
        self
    }

    async fn _make_post_request<S: IntoUrl + Send + Sync>(
        &mut self,
        url: S,
    ) -> Result<reqwest::Response, Box<dyn Error + Send + Sync>> {
        let res = self
            .client
            .post(url)
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&self.config)
            .send()
            .await?;
        Ok(res)
    }

    async fn _make_get_request<S: IntoUrl + Send + Sync>(
        &mut self,
        url: S,
    ) -> Result<reqwest::Response, Box<dyn Error + Send + Sync>> {
        let res = self
            .client
            .get(url)
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await?;
        Ok(res)
    }

    pub async fn models(
        &mut self,
    ) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>> {
        let resp = self._make_get_request(Self::OPENAI_API_MODELS_URL).await?;

        if !resp.status().is_success() {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Error: {}", resp.status()),
            )));
        }

        let data: ModelsResponse = resp.json().await?;
        let model_ids: Vec<String> = data.data.into_iter().map(|model| model.id).collect();
        Ok(model_ids)
    }

    pub async fn check_model(
        &mut self,
        model: &str,
    ) -> Result<Model, Box<dyn std::error::Error + Send + Sync>> {
        let resp = self
            ._make_get_request(format!("{}/{}", Self::OPENAI_API_MODELS_URL, model))
            .await?;

        if !resp.status().is_success() {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Error: {}", resp.status()),
            )));
        }

        let model: Model = resp.json().await?;
        Ok(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_get_all_models() {
        let mut client = OpenAIClient::<Chat>::new();
        let models = client.models().await;
        assert!(models.is_ok());
        assert!(models.unwrap().contains(&"gpt-3.5-turbo".to_string()));
    }

    #[tokio::test]
    async fn test_check_model() {
        let mut client = OpenAIClient::<Chat>::new();
        let model = client.check_model("gpt-3.5-turbo").await;
        assert!(model.is_ok());
    }

    #[tokio::test]
    async fn test_check_model_error() {
        let mut client = OpenAIClient::<Chat>::new();
        let model = client.check_model("gpt-turbo").await;
        assert!(model.is_err());
    }

    #[tokio::test]
    async fn test_single_request() {
        let mut client = OpenAIClient::<Chat>::new().set_stream_responses(false);
        let reply = client.ask("Say this is a test!").await;
        assert_eq!(reply.unwrap(), "This is a test!");
    }

    #[tokio::test]
    async fn test_single_request_streamed() {
        let mut client = OpenAIClient::<Chat>::new();
        let reply = client.ask("Say this is a test!").await;
        assert_eq!(reply.unwrap(), "This is a test!");
    }

    #[tokio::test]
    async fn test_create_single_image_url() {
        let mut client = OpenAIClient::<Image>::new();
        let images = client
            .create_image("A beautiful sunset over the sea.")
            .await;
        assert!(images.is_ok());
        assert_eq!(images.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn test_create_multiple_image_urls() {
        let mut client = OpenAIClient::<Image>::new().set_max_images(2);
        let images = client
            .create_image("A logo for a library written in Rust that deals with AI")
            .await;
        assert!(images.is_ok());
        assert_eq!(images.unwrap().len(), 2);
    }

    #[tokio::test]
    async fn test_create_image_b64_json() {
        let mut client =
            OpenAIClient::<Image>::new().set_response_format(&ImageResponseFormat::Base64Json);
        let images = client
            .create_image("A beautiful sunset over the sea.")
            .await;
        assert!(images.is_ok());
        assert_eq!(images.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn test_image_variation() {
        let mut client = OpenAIClient::<Image>::new();
        let images = client.create_image_variation("./img/logo.png").await;
        assert!(images.is_ok());
        assert_eq!(images.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn test_image_edit() {
        let mut client = OpenAIClient::<Image>::new();
        let images = client
            .edit_image("Make the background transparent", "./img/logo.png", None)
            .await;
        assert!(images.is_ok());
        assert_eq!(images.unwrap().len(), 1);
    }
}
