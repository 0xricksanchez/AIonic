use aionic::openai::chat::Chat;
use aionic::openai::OpenAIClient;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let context_primer = "Answer as if you were Yoda";
    let message = "What is the meaning of life?";

    let mut client = OpenAIClient::<Chat>::new().set_primer(context_primer);

    client
        .ask(message, true) // <-- notice the change here
        .await?;

    client.ask("What did I jsut ask you earlier?", true).await?;
    Ok(())
}
