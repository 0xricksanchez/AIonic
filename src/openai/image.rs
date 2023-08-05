use serde::{Deserialize, Serialize};
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
    pub width: u64,
    pub height: u64,
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
    pub prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mask: Option<String>,
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
