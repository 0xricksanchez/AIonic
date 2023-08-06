use aionic::openai::{Embedding, OpenAIClient};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut client = OpenAIClient::<Embedding>::new();
    let embedding = client
        .embed("The food was delicious and the waiter...")
        .await?;
    println!("{:?}", embedding);
    Ok(())
}
