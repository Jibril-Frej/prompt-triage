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
use model::{Weights, accuracy, balanced_accuracy, train};
use store::{Row, Scored, Store};

/// How many of the most recent labels the running accuracy is computed on.
const RECENT: usize = 50;

/// Prompts longer than this many characters are reported as not trivial
/// without scoring. The embedding model only reads the first 512 tokens
/// (roughly 2000 characters), so a longer prompt would be judged on its
/// beginning alone, and a long request is not a two-minute task anyway.
const LONG_PROMPT: usize = 300;

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

/// Handles one UserPromptSubmit call. Reads the hook JSON on stdin and prints
/// one JSON object for the shell wrapper, or nothing when the prompt should
/// pass through untouched (empty prompt, slash command, or a label reply
/// with nothing pending).
///
/// A normal prompt is scored and kept pending:
/// `{"kind":"predict","verdict":"TRIVIAL","p":0.61,"chars":42}` (`p` is always
/// the probability of "trivial", `chars` the length of the prompt). A prompt
/// longer than `LONG_PROMPT` is neither scored nor kept pending:
/// `{"kind":"predict","verdict":"NOT","p":0.0,"chars":1830,"long":true}`.
///
/// A label reply (`t`, `n`, `trivial` or `not`, any case) labels the pending
/// prompt, refits the classifier and returns that prompt so the wrapper can
/// hand it to the model:
/// `{"kind":"label","label":"TRIVIAL","rows":12,"trivial_rows":5,"recent":12,"accuracy":0.58,"balanced_accuracy":0.55,"prompt":"..."}`.
///
/// A cancel reply (`c` or `cancel`, any case) drops the pending prompt
/// without labeling it and prints `{"kind":"cancel"}`, so the wrapper can
/// block the reply and nothing reaches the model.
fn hook(store: &Store) -> Result<()> {
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input)?;
    let input: serde_json::Value = serde_json::from_str(&input).context("parsing hook input")?;
    let prompt = input["prompt"].as_str().unwrap_or("").trim();
    if prompt.is_empty() || prompt.starts_with('/') {
        return Ok(());
    }
    if is_cancel(prompt) {
        if store.read_pending()?.is_none() {
            return Ok(());
        }
        store.clear_pending()?;
        println!("{}", json!({ "kind": "cancel" }));
        return Ok(());
    }
    if let Some(trivial) = parse_label(prompt) {
        let Some(pending) = store.read_pending()? else {
            return Ok(());
        };
        store.clear_pending()?;
        let text = pending.prompt.clone();
        let report = record(store, pending, trivial)?;
        println!(
            "{}",
            json!({ "kind": "label", "label": label_name(trivial), "rows": report.rows, "trivial_rows": report.trivial_rows, "recent": RECENT.min(report.rows), "accuracy": report.accuracy, "balanced_accuracy": report.balanced_accuracy, "prompt": text })
        );
        return Ok(());
    }
    let chars = prompt.chars().count();
    if chars > LONG_PROMPT {
        println!(
            "{}",
            json!({ "kind": "predict", "verdict": "NOT", "p": 0.0, "chars": chars, "long": true })
        );
        return Ok(());
    }
    let scored = score(store, prompt)?;
    store.write_pending(&scored)?;
    // Convert to f64 before rounding: JSON prints an f32 like 0.67 as 0.6700000166893005.
    let p = (scored.p as f64 * 100.0).round() / 100.0;
    println!(
        "{}",
        json!({ "kind": "predict", "verdict": verdict(scored.p), "p": p, "chars": chars })
    );
    Ok(())
}

/// Recognises a label reply: `t`/`trivial` gives `Some(true)`, `n`/`not`
/// gives `Some(false)`, anything else `None`. Case does not matter.
fn parse_label(prompt: &str) -> Option<bool> {
    match prompt.to_ascii_lowercase().as_str() {
        "t" | "trivial" => Some(true),
        "n" | "not" => Some(false),
        _ => None,
    }
}

/// Recognises a cancel reply: `c` or `cancel`, case does not matter.
fn is_cancel(prompt: &str) -> bool {
    matches!(prompt.to_ascii_lowercase().as_str(), "c" | "cancel")
}

/// Attaches the label to the pending prompt by hand (the hook normally does it).
fn label_pending(store: &Store, trivial: bool) -> Result<()> {
    let scored = store
        .read_pending()?
        .context("no pending prompt to label")?;
    store.clear_pending()?;
    let report = record(store, scored, trivial)?;
    println!(
        "labeled {}; {} rows ({} trivial, {} not); accuracy over last {}: {:.0}% (balanced {:.0}%)",
        label_name(trivial),
        report.rows,
        report.trivial_rows,
        report.rows - report.trivial_rows,
        RECENT.min(report.rows),
        100.0 * report.accuracy,
        100.0 * report.balanced_accuracy
    );
    Ok(())
}

/// What `record` reports after a label: the dataset size, how many rows are
/// trivial, and the accuracy and balanced accuracy over the `RECENT` most
/// recent rows. A struct rather than a tuple so the two counts and the two
/// rates cannot be mixed up at the call site.
struct Report {
    rows: usize,
    trivial_rows: usize,
    accuracy: f32,
    balanced_accuracy: f32,
}

/// Appends the labeled row to the dataset and refits the weights on everything
/// labeled so far.
///
/// Until both classes are present the weights are left as they are (the random
/// ones from setup): a refit on rows of a single class would predict that class
/// for everything with near certainty.
fn record(store: &Store, scored: Scored, trivial: bool) -> Result<Report> {
    store.append_row(&Row {
        scored,
        label: trivial,
    })?;
    let rows = store.read_dataset()?;
    let trivial_rows = rows.iter().filter(|r| r.label).count();
    let both_classes = rows.iter().any(|r| r.label) && rows.iter().any(|r| !r.label);
    if both_classes {
        let examples: Vec<(&[f32], bool)> = rows
            .iter()
            .map(|r| (r.scored.embedding.as_slice(), r.label))
            .collect();
        store.write_weights(&train(&examples, DIM))?;
    }
    // Collected into a Vec because the pairs are walked twice, once per rate.
    let recent: Vec<(f32, bool)> = rows
        .iter()
        .rev()
        .take(RECENT)
        .map(|r| (r.scored.p, r.label))
        .collect();
    Ok(Report {
        rows: rows.len(),
        trivial_rows,
        accuracy: accuracy(recent.iter().copied()),
        balanced_accuracy: balanced_accuracy(recent.iter().copied()),
    })
}

/// Scores `text` and prints the verdict, for trying the model by hand.
fn predict(store: &Store, text: &str) -> Result<()> {
    let scored = score(store, text)?;
    println!("{} {:.2}", verdict(scored.p), scored.p);
    Ok(())
}

/// Prints dataset size, how many rows are trivial, and accuracy and balanced
/// accuracy overall and over the most recent rows.
///
/// Those two lines use the probability each prompt got when it arrived, from
/// a model that had not seen it yet. The "fit" line rescores every row with
/// the current weights instead, so it measures how well the last refit
/// matches its own training data and is close to 100% by construction.
fn stats(store: &Store) -> Result<()> {
    let rows = store.read_dataset()?;
    let trivial = rows.iter().filter(|r| r.label).count();
    let all: Vec<(f32, bool)> = rows.iter().map(|r| (r.scored.p, r.label)).collect();
    let recent = &all[all.len().saturating_sub(RECENT)..];
    let weights = store
        .read_weights()?
        .context("no weights yet: run `triage setup` first")?;
    let fit: Vec<(f32, bool)> = rows
        .iter()
        .map(|r| (weights.predict(&r.scored.embedding), r.label))
        .collect();
    println!(
        "rows: {} ({} trivial, {} not)",
        rows.len(),
        trivial,
        rows.len() - trivial
    );
    println!(
        "accuracy overall: {:.0}% (balanced {:.0}%)",
        100.0 * accuracy(all.iter().copied()),
        100.0 * balanced_accuracy(all.iter().copied())
    );
    println!(
        "accuracy last {}: {:.0}% (balanced {:.0}%)",
        recent.len(),
        100.0 * accuracy(recent.iter().copied()),
        100.0 * balanced_accuracy(recent.iter().copied())
    );
    println!(
        "accuracy of current weights on all data: {:.0}% (balanced {:.0}%)",
        100.0 * accuracy(fit.iter().copied()),
        100.0 * balanced_accuracy(fit.iter().copied())
    );
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
