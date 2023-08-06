use aionic::openai::{Image, OpenAI};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let image_list = OpenAI::<Image>::new()
        .edit("Invert the colors", "./img/logo.png", None)
        .await?;
    println!("Image list: {:?}", image_list);

    Ok(())
}
