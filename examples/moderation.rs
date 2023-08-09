use aionic::openai::{Moderation, OpenAI};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let resp = OpenAI::<Moderation>::new()
        .moderate("I want to kill you.")
        .await?;
    println!("Moderation: {:?}", resp);
    Ok(())
}
