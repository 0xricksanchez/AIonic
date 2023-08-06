use aionic::openai::{Audio, OpenAIClient};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let translate = OpenAIClient::<Audio>::new()
        .translate("examples/samples/colours-german.mp3")
        .await?;
    println!("Translation: {:?}", translate.text);
    Ok(())
}
