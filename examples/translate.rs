use aionic::openai::{Audio, OpenAI};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let translate = OpenAI::<Audio>::new()
        .translate("examples/samples/colours-german.mp3")
        .await?;
    println!("Translation: {:?}", translate.text);
    Ok(())
}
