use aionic::openai::{Image, OpenAI};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let image_prompt = "Create an image that represents the meaning of life?";

    let image_list = OpenAI::<Image>::new().create(image_prompt).await?;
    println!("Image list: {:?}", image_list);

    Ok(())
}
