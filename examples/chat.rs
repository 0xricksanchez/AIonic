use aionic::openai::chat::Chat;
use aionic::openai::OpenAIClient;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let context_primer = "Answer as if you were Yoda";

    OpenAIClient::<Chat>::new()
        .set_primer(context_primer)
        .chat()
        .await?;
    Ok(())
}
