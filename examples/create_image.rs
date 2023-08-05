use aionic::openai::image::Image;
use aionic::openai::OpenAIClient;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let image_prompt = "What is the meaning of life?";

    let image_list = OpenAIClient::<Image>::new()
        .create_image(image_prompt)
        .await?;
    println!("Image list: {:?}", image_list);

    Ok(())
}
