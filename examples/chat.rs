use aionic::openai::{Chat, OpenAI};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let context_primer = "Answer as if you were Yoda";

    OpenAI::<Chat>::new()
        .set_primer(context_primer)
        .chat()
        .await?;
    Ok(())
}
