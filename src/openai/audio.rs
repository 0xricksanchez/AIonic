use serde::{Deserialize, Serialize};
use std::convert::TryFrom;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Response {
    pub text: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(untagged)]
enum FileType {
    Mp3,
    Mp4,
    Mpeg,
    Mpga,
    M4a,
    Wav,
    Webm,
}

impl ToString for FileType {
    fn to_string(&self) -> String {
        match self {
            Self::Mp3 => "mp3".to_string(),
            Self::Mp4 => "mp4".to_string(),
            Self::Mpeg => "mpeg".to_string(),
            Self::Mpga => "mpga".to_string(),
            Self::M4a => "m4a".to_string(),
            Self::Wav => "wav".to_string(),
            Self::Webm => "webm".to_string(),
        }
    }
}

impl TryFrom<&str> for FileType {
    type Error = String; // You can define your own error type here if you prefer.

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "mp3" => Ok(Self::Mp3),
            "mp4" => Ok(Self::Mp4),
            "mpeg" => Ok(Self::Mpeg),
            "mpga" => Ok(Self::Mpga),
            "m4a" => Ok(Self::M4a),
            "wav" => Ok(Self::Wav),
            "webm" => Ok(Self::Webm),
            _ => Err(format!("Invalid file type: {value}")),
        }
    }
}

impl FileType {
    pub fn get_file_type(file: &str) -> Result<Self, String> {
        if let Some(mime_type) = file.split('.').last() {
            Self::try_from(mime_type)
        } else {
            Err(format!("Invalid file type: {file}"))
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(untagged)]
pub enum ResponseFormat {
    Json,
    Text,
    Srt,
    VerboseJson,
    Vtt,
}

impl ToString for ResponseFormat {
    fn to_string(&self) -> String {
        match self {
            Self::Json => "json".to_string(),
            Self::Text => "text".to_string(),
            Self::Srt => "srt".to_string(),
            Self::VerboseJson => "verbose_json".to_string(),
            Self::Vtt => "vtt".to_string(),
        }
    }
}

impl TryFrom<&str> for ResponseFormat {
    type Error = String; // You can define your own error type here if you prefer.

    fn try_from(input: &str) -> Result<Self, Self::Error> {
        match input {
            "json" => Ok(Self::Json),
            "text" => Ok(Self::Text),
            "srt" => Ok(Self::Srt),
            "verbose_json" => Ok(Self::VerboseJson),
            "vtt" => Ok(Self::Vtt),
            _ => Err(format!("Invalid response format: {input}")),
        }
    }
}

impl ResponseFormat {
    pub fn get_response_format(format: &str) -> Result<Self, String> {
        Self::try_from(format)
    }

    pub fn get_default_response_format() -> Self {
        Self::Json
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Audio {
    /// The audio file object (not file name) to transcribe, in one of these formats: mp3, mp4, mpeg, mpga, m4a, wav, or webm.
    pub file: String,

    /// ID of the model to use. Only whisper-1 is currently available.
    pub model: String,

    /// An optional text to guide the model's style or continue a previous audio segment. The prompt should match the audio language.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,

    /// The format of the transcript output, in one of these options: json, text, srt, verbose_json, or vtt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,

    /// The sampling temperature, between 0 and 1. Higher values like 0.8 will make the output more random, while lower
    /// values like 0.2 will make it more focused and deterministic. If set to 0, the model will use log probability to
    /// automatically increase the temperature until certain thresholds are hit.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,

    /// The language of the input audio. Supplying the input language in ISO-639-1 format will improve accuracy and latency.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
}

impl Audio {
    pub const DEFAULT_MODEL: &'static str = "whisper-1";
    pub const ISO_639_1_CODES: &[&'static str] = &[
        "ab", "aa", "af", "ak", "sq", "am", "ar", "an", "hy", "as", "av", "ae", "ay", "az", "bm",
        "ba", "eu", "be", "bn", "bh", "bi", "bs", "br", "bg", "my", "ca", "ch", "ce", "ny", "zh",
        "cv", "kw", "co", "cr", "hr", "cs", "da", "dv", "nl", "dz", "en", "eo", "et", "ee", "fo",
        "fj", "fi", "fr", "ff", "gl", "ka", "de", "el", "gn", "gu", "ht", "ha", "he", "hz", "hi",
        "ho", "hu", "ia", "id", "ie", "ga", "ig", "ik", "io", "is", "it", "iu", "ja", "jv", "kl",
        "kn", "kr", "ks", "kk", "km", "ki", "rw", "ky", "kv", "kg", "ko", "ku", "kj", "la", "lb",
        "lg", "li", "ln", "lo", "lt", "lu", "lv", "gv", "mk", "mg", "ms", "ml", "mt", "mi", "mr",
        "mh", "mn", "na", "nv", "nd", "ne", "ng", "nb", "nn", "no", "ii", "nr", "oc", "oj", "cu",
        "om", "or", "os", "pa", "pi", "fa", "pl", "ps", "pt", "qu", "rm", "rn", "ro", "ru", "sa",
        "sc", "sd", "se", "sm", "sg", "sr", "gd", "sn", "si", "sk", "sl", "so", "st", "es", "su",
        "sw", "ss", "sv", "ta", "te", "th", "ti", "to", "tn", "ts", "tk", "tr", "tw", "ug", "uk",
        "ur", "uz", "ve", "vi", "vo", "wa", "cy", "wo", "fy", "xh", "yi", "yo", "za", "zu",
    ];

    /// Returns the default model to be used by this AI system.
    ///
    /// # Returns
    ///
    /// This function returns a static string slice (`&'static str`) which represents the identifier of the default model used by the AI system.
    pub fn get_default_model() -> &'static str {
        Self::DEFAULT_MODEL
    }

    /// Returns the default model to be used by this AI system.
    ///
    /// # Returns
    ///
    /// This function returns `ResponseFormat` which represents the format of the transcript that is being returned by the AI system.
    pub fn get_default_response_format() -> ResponseFormat {
        ResponseFormat::get_default_response_format()
    }

    pub fn get_response_format(format: &str) -> Result<ResponseFormat, String> {
        ResponseFormat::get_response_format(format)
    }

    pub fn is_file_type_supported(file_name: &str) -> Result<bool, String> {
        let file_type = FileType::get_file_type(file_name)?;
        match file_type {
            FileType::Mp3
            | FileType::Mp4
            | FileType::Mpeg
            | FileType::Mpga
            | FileType::M4a
            | FileType::Wav
            | FileType::Webm => Ok(true),
        }
    }

    pub fn get_supported_file_types() -> Vec<String> {
        vec![
            FileType::Mp3.to_string(),
            FileType::Mp4.to_string(),
            FileType::Mpeg.to_string(),
            FileType::Mpga.to_string(),
            FileType::M4a.to_string(),
            FileType::Wav.to_string(),
            FileType::Webm.to_string(),
        ]
    }

    pub fn get_supported_models() -> Vec<String> {
        vec![Self::DEFAULT_MODEL.to_string()]
    }

    pub fn is_valid_language(language: &str) -> bool {
        if language.len() != 2 {
            return false;
        }
        Self::ISO_639_1_CODES.contains(&language)
    }
}
