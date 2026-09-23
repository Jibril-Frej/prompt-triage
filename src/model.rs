//! The classifier: a logistic regression on top of the frozen embedding.
//!
//! The embedding model turns a prompt into 384 numbers. This module owns the
//! 385 numbers that are actually learned (one weight per dimension plus a
//! bias) and the two things you can do with them: score a prompt, and refit
//! them on the labeled dataset.

use serde::{Deserialize, Serialize};

/// Learning rate of the gradient descent. Embeddings are unit-length, so the
/// gradient is well scaled and a rate of 1 converges without oscillating.
const LEARNING_RATE: f32 = 1.0;

/// Number of full passes over the dataset when refitting. The loss is convex,
/// so more passes only get closer to the same optimum; this is enough to
/// converge on a few hundred rows (see the tests).
const ITERATIONS: usize = 300;

/// Strength of the L2 penalty. It keeps the weights small when the dataset is
/// tiny (a handful of labels would otherwise be fit with huge weights and
/// probabilities of exactly 0 or 1).
const L2: f32 = 1e-3;

/// The learned parameters. `w` has one entry per embedding dimension, `b` is
/// the bias. Plain public fields: this is data, not an object.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Weights {
    pub w: Vec<f32>,
    pub b: f32,
}

impl Weights {
    /// Weights drawn uniformly from [-1, 1], used before the first label so
    /// that the very first verdicts are a coin flip that still depends on the
    /// prompt. They are replaced by `train` as soon as one label exists.
    pub fn random(dim: usize) -> Weights {
        let w = (0..dim).map(|_| rand::random_range(-1.0..1.0)).collect();
        Weights { w, b: 0.0 }
    }

    /// Weights that are all zero; `predict` then returns exactly 0.5.
    /// This is the starting point of every refit.
    pub fn zeros(dim: usize) -> Weights {
        Weights { w: vec![0.0; dim], b: 0.0 }
    }

    /// Probability that `x` is trivial: sigmoid(w · x + b).
    pub fn predict(&self, x: &[f32]) -> f32 {
        let z: f32 = self.w.iter().zip(x).map(|(wi, xi)| wi * xi).sum::<f32>() + self.b;
        sigmoid(z)
    }
}

/// Maps any real number to (0, 1).
fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

/// Refits the weights from scratch on the whole dataset.
///
/// `examples` pairs each embedding with its label (`true` = trivial). The
/// weights start at zero and full-batch gradient descent minimises the
/// average logistic loss plus the L2 penalty. Starting from zero makes the
/// result deterministic: the same dataset always gives the same weights, in
/// whatever order the labels arrived. An empty dataset returns zero weights.
pub fn train(examples: &[(&[f32], bool)], dim: usize) -> Weights {
    let mut weights = Weights::zeros(dim);
    if examples.is_empty() {
        return weights;
    }
    let n = examples.len() as f32;
    let mut grad_w = vec![0.0f32; dim];
    for _ in 0..ITERATIONS {
        grad_w.iter_mut().for_each(|g| *g = 0.0);
        let mut grad_b = 0.0f32;
        for (x, label) in examples {
            let y = if *label { 1.0 } else { 0.0 };
            // d(loss)/dz for the logistic loss is simply (p - y).
            let err = weights.predict(x) - y;
            for (g, xi) in grad_w.iter_mut().zip(*x) {
                *g += err * xi;
            }
            grad_b += err;
        }
        for (wi, g) in weights.w.iter_mut().zip(&grad_w) {
            *wi -= LEARNING_RATE * (g / n + L2 * *wi);
        }
        weights.b -= LEARNING_RATE * grad_b / n;
    }
    weights
}

/// Fraction of `(probability, label)` pairs where the 0.5 threshold agrees
/// with the label. Returns 0 for an empty input.
pub fn accuracy<'a>(pairs: impl IntoIterator<Item = (f32, bool)>) -> f32 {
    let mut total = 0;
    let mut correct = 0;
    for (p, label) in pairs {
        total += 1;
        if (p >= 0.5) == label {
            correct += 1;
        }
    }
    if total == 0 { 0.0 } else { correct as f32 / total as f32 }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// With no training at all the model must be exactly undecided.
    #[test]
    fn zero_weights_give_half() {
        let w = Weights::zeros(4);
        assert_eq!(w.predict(&[0.3, -0.7, 0.1, 0.9]), 0.5);
    }

    /// A dataset where the label is the sign of the first coordinate is
    /// perfectly separable; training must recover that rule.
    #[test]
    fn separable_data_is_learned() {
        let points: Vec<Vec<f32>> = (0..200)
            .map(|i| {
                let t = i as f32 / 200.0;
                let first = if i % 2 == 0 { 0.2 + t } else { -0.2 - t };
                vec![first, t, 1.0 - t]
            })
            .collect();
        let examples: Vec<(&[f32], bool)> =
            points.iter().map(|x| (x.as_slice(), x[0] > 0.0)).collect();
        let weights = train(&examples, 3);
        let acc = accuracy(examples.iter().map(|(x, y)| (weights.predict(x), *y)));
        assert!(acc > 0.95, "accuracy was {acc}");
    }
}
