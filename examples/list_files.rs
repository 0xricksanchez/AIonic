use aionic::openai::{Files, OpenAI};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let res = OpenAI::<Files>::new().list().await?;
    println!("{:#?}", res);
    Ok(())
}
