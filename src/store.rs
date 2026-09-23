//! Everything prompt-triage keeps on disk, all inside one data directory:
//!
//! - `dataset.jsonl`: one JSON object per line, one line per labeled prompt;
//! - `pending.json`: the last scored prompt, waiting for its label;
//! - `weights.json`: the current classifier;
//! - `models/`: the cache of the embedding model.
//!
//! The directory is `$PROMPT_TRIAGE_DIR` if set, otherwise
//! `~/.local/share/prompt-triage` (the XDG data dir on Linux).

use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::model::Weights;

/// A prompt that has been embedded and scored but not labeled yet.
/// `p` is the probability of "trivial" the model gave at that time; it is
/// kept so that accuracy can be computed later without re-scoring.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Scored {
    pub id: String,
    pub ts: u64,
    pub prompt: String,
    pub embedding: Vec<f32>,
    pub p: f32,
}

/// A scored prompt plus the user's label (`true` = trivial).
/// `#[serde(flatten)]` puts the `Scored` fields at the same JSON level as
/// `label`, so a dataset line is one flat object rather than a nested one.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Row {
    #[serde(flatten)]
    pub scored: Scored,
    pub label: bool,
}

/// Handle on the data directory. Every method builds a path under `dir`.
pub struct Store {
    pub dir: PathBuf,
}

impl Store {
    /// Resolves the data directory and creates it if needed.
    pub fn open() -> Result<Store> {
        let dir = match std::env::var_os("PROMPT_TRIAGE_DIR") {
            Some(dir) => PathBuf::from(dir),
            None => dirs::data_dir()
                .context("no data directory on this system")?
                .join("prompt-triage"),
        };
        fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
        Ok(Store { dir })
    }

    /// Where the embedding model is cached.
    pub fn models_dir(&self) -> PathBuf {
        self.dir.join("models")
    }

    /// All labeled rows, oldest first. A missing file is an empty dataset.
    pub fn read_dataset(&self) -> Result<Vec<Row>> {
        let path = self.dir.join("dataset.jsonl");
        let file = match fs::File::open(&path) {
            Ok(file) => file,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(e).with_context(|| format!("opening {}", path.display())),
        };
        let mut rows = Vec::new();
        for (i, line) in BufReader::new(file).lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let row = serde_json::from_str(&line)
                .with_context(|| format!("parsing line {} of {}", i + 1, path.display()))?;
            rows.push(row);
        }
        Ok(rows)
    }

    /// Appends one labeled row to the dataset.
    pub fn append_row(&self, row: &Row) -> Result<()> {
        let path = self.dir.join("dataset.jsonl");
        let mut file = OpenOptions::new().create(true).append(true).open(&path)?;
        let line = serde_json::to_string(row)?;
        writeln!(file, "{line}").with_context(|| format!("appending to {}", path.display()))
    }

    /// Saves the prompt that is waiting for a label, replacing any previous one.
    pub fn write_pending(&self, scored: &Scored) -> Result<()> {
        write_json(self.dir.join("pending.json"), scored)
    }

    /// The prompt waiting for a label, or `None` if there is none.
    pub fn read_pending(&self) -> Result<Option<Scored>> {
        read_json(self.dir.join("pending.json"))
    }

    /// Removes the pending prompt. Nothing to do if there is none.
    pub fn clear_pending(&self) -> Result<()> {
        match fs::remove_file(self.dir.join("pending.json")) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e).context("removing pending.json"),
        }
    }

    /// The current weights, or `None` before the first setup.
    pub fn read_weights(&self) -> Result<Option<Weights>> {
        read_json(self.dir.join("weights.json"))
    }

    /// Replaces the weights file.
    pub fn write_weights(&self, weights: &Weights) -> Result<()> {
        write_json(self.dir.join("weights.json"), weights)
    }
}

/// Deserialises `path`, or returns `None` if the file does not exist.
/// `T: DeserializeOwned` means "a type that can be built from JSON without
/// borrowing from the text", which is what we need since the text is local.
fn read_json<T: serde::de::DeserializeOwned>(path: PathBuf) -> Result<Option<T>> {
    match fs::read_to_string(&path) {
        Ok(text) => Ok(Some(serde_json::from_str(&text)?)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e).with_context(|| format!("reading {}", path.display())),
    }
}

/// Serialises `value` to `path`, overwriting it.
fn write_json<T: Serialize>(path: PathBuf, value: &T) -> Result<()> {
    let text = serde_json::to_string(value)?;
    fs::write(&path, text).with_context(|| format!("writing {}", path.display()))
}
