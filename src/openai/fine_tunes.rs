use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct EventResponse {
    pub object: String,
    pub data: Vec<Event>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ListResponse {
    pub data: Option<Vec<ListSummary>>,
    pub object: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ListSummary {
    pub id: String,
    pub object: String,
    pub bytes: u64,
    pub created_at: u64,
    pub filename: String,
    pub purpose: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Response {
    pub id: String,
    pub object: String,
    pub model: String,
    pub created_at: u64,
    pub events: Vec<Event>,
    pub fine_tuned_model: Option<String>,
    pub hyperparams: HyperParams,
    pub organization_id: String,
    pub result_files: Vec<File>,
    pub status: String,
    pub validation_files: Vec<String>,
    pub training_files: Vec<File>,
    pub updated_at: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Event {
    pub object: String,
    pub created_at: u64,
    pub level: String,
    pub message: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HyperParams {
    pub batch_size: u64,
    pub learning_rate_multiplier: f64,
    pub n_epochs: u64,
    pub prompt_loss_weight: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct File {
    pub id: String,
    pub object: String,
    pub bytes: u64,
    pub created_at: u64,
    pub filename: String,
    pub purpose: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FineTune {
    /// Creates a job that fine-tunes a specified model from a given dataset.
    pub training_file: String,

    /// The ID of an uploaded file that contains validation data.
    ///
    /// If you provide this file, the data is used to generate validation metrics
    /// periodically during fine-tuning. These metrics can be viewed in the fine-tuning
    /// results file. Your train and validation data should be mutually exclusive.
    ///
    /// Your dataset must be formatted as a JSONL file, where each validation example is
    /// a JSON object with the keys "prompt" and "completion". Additionally, you must
    /// upload your file with the purpose "fine-tune".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub validation_file: Option<String>,

    /// The name of the base model to fine-tune.
    /// You can select one of "ada", "babbage", "curie", "davinci", or a fine-tuned
    /// model created after 2022-04-21.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    /// The number of epochs to train the model for. An epoch refers to one full cycle
    /// through the training dataset.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_epochs: Option<u64>,

    /// The batch size to use for training. The batch size is the number of training examples
    /// used to train a single forward and backward pass.
    ///
    /// By default, the batch size will be dynamically configured to be ~0.2% of the number
    /// of examples in the training set, capped at 256 - in general, we've found that larger
    /// batch sizes tend to work better for larger datasets.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub batch_size: Option<u64>,

    /// The learning rate multiplier to use for training. The fine-tuning learning rate is the
    /// original learning rate used for pretraining multiplied by this value.
    ///
    /// By default, the learning rate multiplier is the 0.05, 0.1, or 0.2 depending on final `batch_size`
    /// (larger learning rates tend to perform better with larger batch sizes).
    /// We recommend experimenting with values in the range 0.02 to 0.2 to see what produces the best results.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub learning_rate_multiplier: Option<f64>,

    /// The weight to use for loss on the prompt tokens. This controls how much the model tries to
    /// learn to generate the prompt (as compared to the completion which always has a weight of 1.0),
    /// and can add a stabilizing effect to training when completions are short.
    ///
    /// If prompts are extremely long (relative to completions), it may make sense to reduce this weight so
    /// as to avoid over-prioritizing learning the prompt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_loss_weight: Option<f64>,

    /// If set, we calculate classification-specific metrics such as accuracy and F-1 score using the
    /// validation set at the end of every epoch. These metrics can be viewed in the results file.
    ///
    /// In order to compute classification metrics, you must provide a validation_file. Additionally,
    /// you must specify `classification_n_classes` for multiclass classification or `classification_positive_class`
    /// for binary classification.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compute_classification_metrics: Option<bool>,

    /// The number of classes in a classification task.
    ///
    /// This parameter is required for multiclass classification.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub classification_n_classes: Option<u64>,

    /// The positive class in binary classification.
    ///
    /// This parameter is needed to generate precision, recall, and F1 metrics when doing binary classification.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub classification_positive_class: Option<String>,

    /// If this is provided, we calculate F-beta scores at the specified beta values. The F-beta score is a
    /// generalization of F-1 score. This is only used for binary classification.
    ///
    /// With a beta of 1 (i.e. the F-1 score), precision and recall are given the same weight. A larger beta
    /// score puts more weight on recall and less on precision. A smaller beta score puts more weight on precision
    /// and less on recall.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub classification_betas: Option<Vec<f64>>,

    /// A string of up to 40 characters that will be added to your fine-tuned model name.
    ///
    /// For example, a suffix of "custom-model-name" would produce a model name like
    /// ada:ft-your-org:custom-model-name-2022-02-15-04-21-04.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suffix: Option<String>,
}

impl FineTune {
    pub fn set_model(&mut self, model: String) {
        self.model = Some(model);
    }

    pub fn get_default_model() -> &'static str {
        "curie"
    }

    pub fn get_default_n_epochs() -> u64 {
        4
    }

    pub fn get_default_prompt_loss_weight() -> f64 {
        0.01
    }

    pub fn get_default_compute_classification_metrics() -> bool {
        false
    }
}
