use aionic::openai::{Chat, OpenAI};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let context_primer = "Answer as if you were Yoda";
    let message = "What is the meaning of life?";

    let mut client = OpenAI::<Chat>::new().set_primer(context_primer);

    client
        .ask(message, true) // <-- notice the change here
        .await?;

    client.ask("What did I just ask you earlier?", true).await?;
    Ok(())
}
