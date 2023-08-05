use aionic::openai::image::Image;
use aionic::openai::OpenAIClient;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let image_list = OpenAIClient::<Image>::new()
        .edit_image("Invert the colors", "./img/logo.png", None)
        .await?;
    println!("Image list: {:?}", image_list);

    Ok(())
}
