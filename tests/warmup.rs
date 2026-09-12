// Build-time only. Pulls the pinned checkpoint into the hf-hub cache so the verifier can run
// the agent's program with the network closed. Writing the cache with the same library the
// agent's crate reads it with is deliberate: it guarantees the on-disk layout is one hf-hub
// resolves without a request.
use anyhow::Result;
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};

fn main() -> Result<()> {
    let repo = ApiBuilder::new().build()?.repo(Repo::with_revision(
        "sentence-transformers/all-MiniLM-L6-v2".to_string(),
        RepoType::Model,
        "1110a243fdf4706b3f48f1d95db1a4f5529b4d41".to_string(),
    ));
    for f in ["config.json", "tokenizer.json", "model.safetensors"] {
        println!("cached {}", repo.get(f)?.display());
    }
    Ok(())
}
