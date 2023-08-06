use aionic::openai::{Audio, OpenAI};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let transcribe = OpenAI::<Audio>::new()
        .transcribe("examples/samples/sample-1.mp3")
        .await?;
    println!("Transcription: {:?}", transcribe.text);
    Ok(())
}
