use serde::{Deserialize, Serialize};

/// Enum representing the format in which the response from OpenAI's Image API can be received.
///
/// This can either be a URL (Url) pointing to the generated image, or a Base64-encoded JSON string (Base64Json)
/// that represents the image.
pub enum ResponseDataType {
    Url,
    Base64Json,
}

impl ToString for ResponseDataType {
    fn to_string(&self) -> String {
        match self {
            Self::Url => "url".to_string(),
            Self::Base64Json => "b64_json".to_string(),
        }
    }
}

/// Struct representing the size of an image.
///
/// It consists of the width and the height of the image, both represented as unsigned 64-bit integers.
#[derive(Clone, Debug, Copy)]
pub struct Size {
    /// The width of the image in pixels.
    pub width: u64,

    /// The height of the image in pixels.
    pub height: u64,
}

impl Size {
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

/// Represents the response from an Image API call to OpenAI.
///
/// Contains fields that provide information about the creation time and the data associated with the generated image.
#[derive(Deserialize, Debug)]
pub struct Response {
    /// UNIX timestamp indicating when the image was created.
    pub created: u64,

    /// A vector of ImageData objects, each representing a generated image.
    pub data: Vec<ImageData>,
}

/// Represents the data associated with a single generated image in an Image API response.
/// Only ever one of the fields is present in a single ImageData object.
#[derive(Deserialize, Debug)]
pub struct ImageData {
    /// The URL of the generated image. This field is present when the response format is set to Url.
    pub url: Option<String>,

    /// A Base64-encoded JSON string representing the generated image. This field is present when the response format is set to Base64Json.
    pub b64_json: Option<String>,
}

impl ToString for Size {
    fn to_string(&self) -> String {
        format!("{}x{}", self.width, self.height)
    }
}

/// Represents an Image object in the OpenAI Image API.
///
/// This struct includes fields like `prompt`, `n`, `size`, `response_format`, `user`, `image`, and `mask`.
/// Each of these fields can be set according to the requirements of the Image API request.
/// Optional fields are represented as `Option<T>`.
///
/// For more information check the official [openAI API documentation](https://platform.openai.com/docs/api-reference/images)
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Image {
    /// A text description of the desired image(s). The maximum length is 1000 characters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,

    /// The number of images to generate. Must be between 1 and 10.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u64>,

    /// The size of the generated images. Must be one of 256x256, 512x512, or 1024x1024.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<String>,

    /// The format in which the generated images are returned. Must be of type `ImageResponseFormat`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,

    /// A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,

    /// Note: Only for edit/variation requests!
    ///
    /// The image to edit. Must be a valid PNG file, less than 4MB, and square.
    /// If mask is not provided, image must have transparency, which will be used as the mask.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image: Option<String>,

    /// Note: Only for edit requests!
    ///
    /// An additional image whose fully transparent areas (e.g. where alpha is zero) indicate where
    /// image should be edited. Must be a valid PNG file, less than 4MB, and have the same dimensions as image.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mask: Option<String>,
}

impl Image {
    const DEFAULT_N: u64 = 1;
    const DEFAULT_SIZE: &str = "1024x1024";
    const DEFAULT_RESPONSE_FORMAT: &str = "url";

    /// Returns the default n for the Image API.
    ///
    /// # Returns
    ///
    /// This function returns a `u64` value which represents the default n.
    pub fn get_default_n() -> u64 {
        Self::DEFAULT_N
    }

    /// Returns the default image size for the Image API.
    ///
    /// # Returns
    ///
    /// This function returns a `&'static str` value which represents the default size.
    pub fn get_default_size() -> &'static str {
        Self::DEFAULT_SIZE
    }

    /// Returns the default response data type for the Image API.
    ///
    /// # Returns
    ///
    /// This function returns a `&'static str` value which represents the default response data type.
    pub fn get_default_response_format() -> &'static str {
        Self::DEFAULT_RESPONSE_FORMAT
    }

    /// Checks if the current Image object is valid in terms of its size fields
    ///
    /// # Returns
    ///
    /// This function returns a `bool` value which represents whether the Image size is valid.
    pub fn is_valid_size(size: &str) -> bool {
        let valid_sizes = ["256x256", "512x512", "1024x1024"];
        valid_sizes.contains(&size)
    }

    /// Checks if the current Image object is valid in terms of the requested response format
    ///
    /// # Returns
    ///
    /// This function returns a `bool` value which represents whether the Image response format is valid.
    pub fn is_valid_response_format(response_format: &str) -> bool {
        let valid_response_formats = ["url", "b64_json"];
        valid_response_formats.contains(&response_format)
    }

    /// Checks if the current Image object is valid in terms of its n field
    ///
    /// # Returns
    ///
    /// This function returns a `bool` value which represents whether the Image n is valid.
    pub fn is_valid_n(n: u64) -> bool {
        (1..=10).contains(&n)
    }
}
