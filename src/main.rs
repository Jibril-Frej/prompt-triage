//! `triage`: a local classifier that guesses whether a prompt to a coding
//! agent is trivial, and learns from the user's answer to that guess.
//!
//! The commands are wired here; the work is in the three modules:
//! `embed` (prompt -> vector), `model` (vector -> probability, and training),
//! `store` (files on disk).

mod embed;
mod model;
mod store;

use std::io::Read;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use serde_json::json;

use embed::{DIM, Embedder};
use model::{Weights, accuracy, train};
use store::{Row, Scored, Store};

/// How many of the most recent labels the running accuracy is computed on.
const RECENT: usize = 50;

#[derive(Parser)]
#[command(about = "Local trivial-or-not classifier for coding-agent prompts")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Download the embedding model and create the initial random weights.
    Setup,
    /// Read a UserPromptSubmit JSON on stdin, score the prompt, keep it pending.
    Hook,
    /// Label the pending prompt and refit the classifier.
    Label { label: Label },
    /// Score an arbitrary text without keeping it pending.
    Predict { text: String },
    /// Dataset size, class balance and accuracy.
    Stats,
}

/// The two possible labels, as typed on the command line.
#[derive(ValueEnum, Clone, Copy, PartialEq)]
enum Label {
    Trivial,
    Not,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let store = Store::open()?;
    match cli.command {
        Command::Setup => setup(&store),
        Command::Hook => hook(&store),
        Command::Label { label } => label_pending(&store, label == Label::Trivial),
        Command::Predict { text } => predict(&store, &text),
        Command::Stats => stats(&store),
    }
}

/// Downloads the model (with a progress bar) and writes random weights if
/// there are none yet, so that the hook never has to do either.
fn setup(store: &Store) -> Result<()> {
    Embedder::load(&store.models_dir(), true)?;
    if store.read_weights()?.is_none() {
        store.write_weights(&Weights::random(DIM))?;
    }
    println!("ready: data in {}", store.dir.display());
    Ok(())
}

/// Handles one hook call. Reads the UserPromptSubmit JSON on stdin and prints
/// one JSON object for the shell wrapper, or nothing when the prompt should
/// pass through untouched (empty prompt or slash command).
///
/// A plain prompt is scored and kept pending:
/// `{"kind":"predict","verdict":"TRIVIAL","p":0.61}`.
/// A prompt starting with `t:` or `n:` carries a label. It is recorded for the
/// pending prompt if the text matches, otherwise for the text itself, and the
/// classifier is refitted:
/// `{"kind":"label","label":"TRIVIAL","rows":12,"recent":12,"accuracy":0.58}`.
fn hook(store: &Store) -> Result<()> {
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input)?;
    let input: serde_json::Value = serde_json::from_str(&input).context("parsing hook input")?;
    let prompt = input["prompt"].as_str().unwrap_or("").trim();
    if prompt.is_empty() || prompt.starts_with('/') {
        return Ok(());
    }
    let Some((trivial, text)) = strip_label(prompt) else {
        let scored = score(store, prompt)?;
        store.write_pending(&scored)?;
        // Convert to f64 before rounding: JSON prints an f32 like 0.67 as 0.6700000166893005.
        let p = (scored.p as f64 * 100.0).round() / 100.0;
        println!("{}", json!({ "kind": "predict", "verdict": verdict(scored.p), "p": p }));
        return Ok(());
    };
    if text.is_empty() {
        return Ok(());
    }
    let scored = match store.read_pending()? {
        Some(pending) if pending.prompt == text => pending,
        _ => score(store, text)?,
    };
    store.clear_pending()?;
    let (rows, acc) = record(store, scored, trivial)?;
    println!(
        "{}",
        json!({ "kind": "label", "label": label_name(trivial), "rows": rows, "recent": RECENT.min(rows), "accuracy": acc })
    );
    Ok(())
}

/// Splits a `t:` or `n:` prefix (any case) off a prompt. Returns the label
/// (`true` = trivial) and the rest of the prompt, trimmed.
fn strip_label(prompt: &str) -> Option<(bool, &str)> {
    let (head, rest) = prompt.split_at_checked(2)?;
    let trivial = match head.to_ascii_lowercase().as_str() {
        "t:" => true,
        "n:" => false,
        _ => return None,
    };
    Some((trivial, rest.trim()))
}

/// Attaches the label to the pending prompt by hand (the hook normally does it).
fn label_pending(store: &Store, trivial: bool) -> Result<()> {
    let scored = store.read_pending()?.context("no pending prompt to label")?;
    store.clear_pending()?;
    let (rows, acc) = record(store, scored, trivial)?;
    println!(
        "labeled {}; {} rows; accuracy over last {}: {:.0}%",
        label_name(trivial),
        rows,
        RECENT.min(rows),
        100.0 * acc
    );
    Ok(())
}

/// Appends the labeled row to the dataset and refits the weights on everything
/// labeled so far. Returns the dataset size and the accuracy over the most
/// recent rows.
fn record(store: &Store, scored: Scored, trivial: bool) -> Result<(usize, f32)> {
    store.append_row(&Row { scored, label: trivial })?;
    let rows = store.read_dataset()?;
    let examples: Vec<(&[f32], bool)> =
        rows.iter().map(|r| (r.scored.embedding.as_slice(), r.label)).collect();
    store.write_weights(&train(&examples, DIM))?;
    let recent = rows.iter().rev().take(RECENT).map(|r| (r.scored.p, r.label));
    Ok((rows.len(), accuracy(recent)))
}

/// Scores `text` and prints the verdict, for trying the model by hand.
fn predict(store: &Store, text: &str) -> Result<()> {
    let scored = score(store, text)?;
    println!("{} {:.2}", verdict(scored.p), scored.p);
    Ok(())
}

/// Prints dataset size, how many rows are trivial, and accuracy overall and
/// over the most recent rows.
fn stats(store: &Store) -> Result<()> {
    let rows = store.read_dataset()?;
    let trivial = rows.iter().filter(|r| r.label).count();
    let all = rows.iter().map(|r| (r.scored.p, r.label));
    let recent = rows.iter().rev().take(RECENT).map(|r| (r.scored.p, r.label));
    println!("rows: {} ({} trivial, {} not)", rows.len(), trivial, rows.len() - trivial);
    println!("accuracy overall: {:.0}%", 100.0 * accuracy(all));
    println!("accuracy last {}: {:.0}%", RECENT.min(rows.len()), 100.0 * accuracy(recent));
    Ok(())
}

/// Embeds `text` and scores it with the current weights.
fn score(store: &Store, text: &str) -> Result<Scored> {
    let weights = store
        .read_weights()?
        .context("no weights yet: run `triage setup` first")?;
    let mut embedder = Embedder::load(&store.models_dir(), false)?;
    let embedding = embedder.embed(text)?;
    Ok(Scored {
        id: format!("{:x}", rand::random::<u64>()),
        ts: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
        prompt: text.to_string(),
        p: weights.predict(&embedding),
        embedding,
    })
}

/// The decision at the 0.5 threshold.
fn verdict(p: f32) -> &'static str {
    label_name(p >= 0.5)
}

/// The label as it is printed and stored in the hook output.
fn label_name(trivial: bool) -> &'static str {
    if trivial { "TRIVIAL" } else { "NOT" }
}
