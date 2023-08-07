use aionic::openai::{Files, OpenAI};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut client = OpenAI::<Files>::new();
    let res = client.list().await?;
    println!("current uploads: {:#?}", res);

    let res = client.upload("examples/samples/test.jsonl").await?;
    println!("uploaded: {:#?}", res);

    println!("waiting for file to be processed...");
    // sleep for 3 seconds to allow the file to be processed
    tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;

    let res = client.list().await?;
    println!("current uploads: {:#?}", res);

    let res = client.delete(res.data[0].id.clone()).await?;
    println!("deleted: {:#?}", res);
    Ok(())
}
