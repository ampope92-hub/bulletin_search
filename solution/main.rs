// Reference solution. Splits every issue on its own item boundaries, embeds each item with
// all-MiniLM-L6-v2, and scores an issue by its single best-matching item.
//
// The granularity is the whole point. An issue runs to roughly 900 tokens, well past the
// checkpoint's 256-token limit, and holds sixteen unrelated items; mean pooling over a
// window that spans several of them averages the one item a query is about together with
// fifteen it is not, and the match disappears into the average. Splitting on the item
// headings keeps each embedding about exactly one thing.
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

const MODEL_ID: &str = "sentence-transformers/all-MiniLM-L6-v2";
const REVISION: &str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41";
const CORPUS_DIR: &str = "/app/corpus";
const QUERIES: &str = "/app/queries/graded.json";
const OUTPUT: &str = "/app/rankings.json";
const MAX_TOKENS: usize = 256;
const TOP_K: usize = 10;
const BATCH: usize = 32;
// The index unit, and the one decision this task turns on. `true` embeds each item on its
// own; `false` embeds the whole issue as a single span, which is what a pipeline that never
// looks at how an issue is put together does. verify_arms.sh flips it for the
// oracle_whole_issue arm, which has to score 0 for the bar to be measuring anything.
const INDEX_BY_ITEM: bool = true;
const SEP: u32 = 102;

#[derive(Deserialize)]
struct Query {
    query_id: String,
    query: String,
}

#[derive(Serialize)]
struct Ranking {
    query_id: String,
    issues: Vec<String>,
}

struct Encoder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl Encoder {
    fn load() -> Result<Self> {
        // A pinned revision means the cache key is a commit sha, so once the files are on
        // disk this resolves without touching the network at all.
        let api = ApiBuilder::new().build()?;
        let repo = api.repo(Repo::with_revision(
            MODEL_ID.to_string(),
            RepoType::Model,
            REVISION.to_string(),
        ));
        let config_path = repo.get("config.json")?;
        let tokenizer_path = repo.get("tokenizer.json")?;
        let weights_path = repo.get("model.safetensors")?;

        let config: Config = serde_json::from_str(&fs::read_to_string(config_path)?)?;
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;
        let device = Device::Cpu;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)?
        };
        let model = BertModel::load(vb, &config)?;
        Ok(Self { model, tokenizer, device })
    }

    fn token_ids(&self, text: &str) -> Result<Vec<u32>> {
        let encoded = self
            .tokenizer
            .encode(text, true)
            .map_err(anyhow::Error::msg)?;
        let mut ids: Vec<u32> = encoded.get_ids().to_vec();
        if ids.len() > MAX_TOKENS {
            ids.truncate(MAX_TOKENS - 1);
            ids.push(SEP);
        }
        Ok(ids)
    }

    /// Mean pooling over the token embeddings, then L2 normalisation -- what the
    /// sentence-transformers checkpoint was trained to produce.
    ///
    /// Sequences are encoded in batches, which means padding, which means the pooling has to
    /// be masked: averaging over the padded positions as well would make an item's vector
    /// depend on whatever else happened to share its batch. Batching by similar length keeps
    /// the padding small; on this corpus it is the difference between a few minutes and
    /// most of an hour.
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut encoded: Vec<(usize, Vec<u32>)> = texts
            .iter()
            .map(|t| self.token_ids(t))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .enumerate()
            .collect();
        encoded.sort_by_key(|(_, ids)| ids.len());

        let mut out = vec![Vec::new(); texts.len()];
        for chunk in encoded.chunks(BATCH) {
            let width = chunk.iter().map(|(_, ids)| ids.len()).max().unwrap_or(1);
            let rows = chunk.len();
            let mut flat_ids = vec![0u32; rows * width];
            let mut flat_mask = vec![0u32; rows * width];
            for (r, (_, ids)) in chunk.iter().enumerate() {
                for (c, id) in ids.iter().enumerate() {
                    flat_ids[r * width + c] = *id;
                    flat_mask[r * width + c] = 1;
                }
            }
            let input = Tensor::from_vec(flat_ids, (rows, width), &self.device)?;
            let mask = Tensor::from_vec(flat_mask, (rows, width), &self.device)?;
            let types = Tensor::zeros((rows, width), DType::U32, &self.device)?;
            let hidden = self.model.forward(&input, &types, Some(&mask))?;

            let maskf = mask.to_dtype(DType::F32)?.unsqueeze(2)?;
            let summed = hidden.broadcast_mul(&maskf)?.sum(1)?;
            let counts = maskf.sum(1)?.clamp(1e-9f32, f32::INFINITY)?;
            let pooled = summed.broadcast_div(&counts)?.to_vec2::<f32>()?;

            for (r, (original, _)) in chunk.iter().enumerate() {
                let v = &pooled[r];
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
                out[*original] = v.iter().map(|x| x / norm).collect();
            }
        }
        Ok(out)
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        Ok(self.embed_batch(&[text.to_string()])?.remove(0))
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Every issue is a masthead line followed by items separated by blank lines. Dropping the
/// first block leaves one string per item.
fn split_items(text: &str) -> Vec<String> {
    if !INDEX_BY_ITEM {
        return vec![text.to_string()];
    }
    text.split("\n\n")
        .skip(1)
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

fn corpus_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut paths: Vec<PathBuf> = fs::read_dir(dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "md").unwrap_or(false))
        .collect();
    paths.sort();
    Ok(paths)
}

fn main() -> Result<()> {
    let encoder = Encoder::load()?;

    let mut issue_ids: Vec<String> = Vec::new();
    let item_vectors: Vec<Vec<f32>>;
    let mut pending: Vec<String> = Vec::new();
    let mut item_owner: Vec<usize> = Vec::new();

    for path in corpus_files(Path::new(CORPUS_DIR))? {
        let id = path
            .file_stem()
            .and_then(|s| s.to_str())
            .context("issue file has no stem")?
            .to_string();
        let text = fs::read_to_string(&path)?;
        let owner = issue_ids.len();
        for item in split_items(&text) {
            pending.push(item);
            item_owner.push(owner);
        }
        issue_ids.push(id);
    }
    item_vectors = encoder.embed_batch(&pending)?;
    eprintln!("indexed {} items across {} issues", item_vectors.len(), issue_ids.len());

    let queries: Vec<Query> = serde_json::from_str(&fs::read_to_string(QUERIES)?)?;
    let mut out: Vec<Ranking> = Vec::with_capacity(queries.len());

    for q in &queries {
        let qv = encoder.embed(&q.query)?;
        let mut best: HashMap<usize, f32> = HashMap::new();
        for (vec, owner) in item_vectors.iter().zip(&item_owner) {
            let score = dot(&qv, vec);
            let slot = best.entry(*owner).or_insert(f32::NEG_INFINITY);
            if score > *slot {
                *slot = score;
            }
        }
        let mut ranked: Vec<(usize, f32)> = best.into_iter().collect();
        // Ties broken by issue order so the ranking is reproducible run to run.
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
        out.push(Ranking {
            query_id: q.query_id.clone(),
            issues: ranked
                .iter()
                .take(TOP_K)
                .map(|(i, _)| issue_ids[*i].clone())
                .collect(),
        });
    }

    fs::write(OUTPUT, serde_json::to_string_pretty(&out)? + "\n")?;
    eprintln!("wrote {} rankings to {}", out.len(), OUTPUT);
    Ok(())
}
