use aionic::openai::chat_completion::Chat;
use aionic::openai::OpenAIClient;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let context_primer = "Answer as if you were Yoda";
    let message = "What is the meaning of life?";

    let _res = OpenAIClient::<Chat>::new()
        .set_model(Chat::get_default_model())
        .set_temperature(Chat::get_default_temperature())
        .set_max_tokens(Chat::get_default_max_tokens())
        .set_stream_responses(Chat::get_default_stream())
        .set_primer(context_primer)
        .ask(message, false)
        .await?;
    Ok(())
}
