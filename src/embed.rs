//! Turns a prompt into a fixed-size vector with a small sentence-embedding
//! model (all-MiniLM-L6-v2, run locally through ONNX by the `fastembed` crate).
//!
//! The model is never trained here: it is a frozen feature extractor. The
//! learning happens in `model.rs`, on top of these vectors.

use std::path::Path;

use anyhow::{Context, Result};
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};

/// Length of an all-MiniLM-L6-v2 embedding.
pub const DIM: usize = 384;

/// A loaded embedding model. Loading takes a noticeable fraction of a second
/// (ONNX runtime plus a ~90 MB model), so each command loads it at most once.
pub struct Embedder {
    model: TextEmbedding,
}

impl Embedder {
    /// Loads the model from `cache_dir`, downloading it from Hugging Face on
    /// first use. `show_download_progress` prints a progress bar on stderr;
    /// keep it off inside the hook, where stderr is not a terminal.
    pub fn load(cache_dir: &Path, show_download_progress: bool) -> Result<Embedder> {
        let options = TextInitOptions::new(EmbeddingModel::AllMiniLML6V2)
            .with_cache_dir(cache_dir.to_path_buf())
            .with_show_download_progress(show_download_progress);
        let model = TextEmbedding::try_new(options).context("loading the embedding model")?;
        Ok(Embedder { model })
    }

    /// Embeds one text. `fastembed` works on batches, so this wraps the text in
    /// a one-element batch and takes the single result back out.
    pub fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
        let mut vectors = self.model.embed([text], None).context("embedding the prompt")?;
        vectors.pop().context("the embedding model returned nothing")
    }
}
