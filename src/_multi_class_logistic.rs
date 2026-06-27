use ndarray::{azip, Array2, ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;

// ── Primitives ────────────────────────────────────────────────────────────────

/// In-place numerically stable softmax.
#[inline(always)]
pub fn softmax_inplace(v: &mut [f32]) {
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in v.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    let inv = sum.recip();
    for x in v.iter_mut() {
        *x *= inv;
    }
}

/// Flat dot product over a 2D array treated as a contiguous buffer.
/// Both arrays must have the same shape and be in standard (row-major) layout.
#[inline(always)]
pub fn dot2d(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// In-place scaled addition: a += alpha * b  (elementwise, same shape).
#[inline(always)]
pub fn scaled_add2d(a: &mut Array2<f32>, alpha: f32, b: &Array2<f32>) {
    azip!((x in a, &y in b) *x += alpha * y);
}

// ── Fused loss + gradient ─────────────────────────────────────────────────────

/// Single pass over x that computes both cross-entropy loss and its gradient.
///
/// * `w`  – `[n_classes, n_features]`
/// * returns `(loss, grad)` where grad has the same shape as `w`
pub fn compute_loss_and_gradient_multiclass(
    x: ArrayView2<f32>,
    y: ArrayView1<usize>,
    w: ArrayView2<f32>,
    l1_reg: f32,
    l2_reg: f32,
    sample_weights_sum: f32,
    sample_weights: Option<ArrayView1<f32>>,
) -> (f32, Array2<f32>) {
    let n_classes  = w.nrows();
    let n_features = w.ncols();
    let flat_len   = n_classes * n_features;

    // left arm:  one pass over rows of x → (log_loss scalar, flat grad vec)
    // right arm: one pass over w         → (reg_loss scalar, flat reg_grad vec)
    // both arms run in parallel
    let ((log_loss, loss_grad), (reg_loss, reg_grad)) = rayon::join(
        || {
            x.axis_iter(Axis(0))
                .into_par_iter()
                .enumerate()
                .fold(
                    || (0.0f32, vec![0.0f32; flat_len]),
                    |mut acc, (i, row)| {
                        // logits = w · xᵢ — one dot per class
                        let mut logits: Vec<f32> = w
                            .outer_iter()
                            .map(|wk| row.dot(&wk))
                            .collect();

                        softmax_inplace(&mut logits);

                        let weight = sample_weights.map_or(1.0, |sw| sw[i]);

                        // loss: -log p(y=yᵢ | xᵢ), weighted
                        // read before we subtract 1 from logits[y[i]]
                        acc.0 -= logits[y[i]].ln() * weight;

                        // residual in-place: softmax - one_hot(y[i])
                        logits[y[i]] -= 1.0;

                        // outer product residual ⊗ xᵢ accumulated into flat grad
                        // acc[k * n_features + j] += weight * residual[k] * xᵢ[j]
                        let row_sl = row.as_slice().unwrap();
                        for (k, &res) in logits.iter().enumerate() {
                            let scaled = weight * res;
                            // skip zero residuals (true class with weight=1 after
                            // subtracting 1 can still be non-zero, but other classes
                            // with softmax ≈ 0 hit this branch)
                            if scaled == 0.0 { continue; }
                            let base = k * n_features;
                            // tight inner loop — compiler will auto-vectorise
                            for (j, &xj) in row_sl.iter().enumerate() {
                                acc.1[base + j] += scaled * xj;
                            }
                        }
                        acc
                    },
                )
                .reduce(
                    || (0.0f32, vec![0.0f32; flat_len]),
                    |mut a, b| {
                        a.0 += b.0;
                        // elementwise reduction of per-thread grad accumulators
                        a.1.iter_mut()
                            .zip(b.1.iter())
                            .for_each(|(ai, &bi)| *ai += bi);
                        a
                    },
                )
        },
        || {
            // reg loss and grad share a single pass over w
            // w is [n_classes * n_features] — tiny vs x, two passes is fine
            // but we do it in one for clarity
            let (rl, rg): (Vec<f32>, Vec<f32>) = w
                .into_par_iter()
                .map(|&wi| (
                    l1_reg * wi.abs() + 0.5 * l2_reg * wi * wi,
                    l1_reg * wi.signum() + l2_reg * wi,
                ))
                .unzip();
            // unzip gives (Vec<f32>, Vec<f32>) — sum the loss half
            let reg_loss: f32 = rl.iter().sum();
            (reg_loss, rg)
        },
    );

    // scale and combine into a single Array2 — one allocation, one pass
    let inv = sample_weights_sum.recip();
    let combined: Vec<f32> = loss_grad
        .iter()
        .zip(reg_grad.iter())
        .map(|(&lg, &rg)| (lg + rg) * inv)
        .collect();

    let total_loss = (log_loss + reg_loss) * inv;
    let grad = Array2::from_shape_vec((n_classes, n_features), combined)
        .expect("shape mismatch in fused kernel");

    (total_loss, grad)
}

// ── Standalone loss (for Armijo check — no gradient) ─────────────────────────

/// Loss only — cheaper than the fused kernel; used for rejected line-search steps.
pub fn compute_loss_multiclass(
    x: ArrayView2<f32>,
    y: ArrayView1<usize>,
    w: ArrayView2<f32>,
    l1_reg: f32,
    l2_reg: f32,
    sample_weights_sum: f32,
    sample_weights: Option<ArrayView1<f32>>,
) -> f32 {
    let (log_loss, reg) = rayon::join(
        || {
            x.axis_iter(Axis(0))
                .into_par_iter()
                .enumerate()
                .map(|(i, row)| {
                    let mut logits: Vec<f32> = w
                        .outer_iter()
                        .map(|wk| row.dot(&wk))
                        .collect();
                    softmax_inplace(&mut logits);
                    let ce = -logits[y[i]].ln();
                    sample_weights.map_or(ce, |sw| ce * sw[i])
                })
                .sum::<f32>()
        },
        || {
            w.iter()
                .map(|&wi| l1_reg * wi.abs() + 0.5 * l2_reg * wi * wi)
                .sum::<f32>()
        },
    );
    (log_loss + reg) / sample_weights_sum
}

// ── Predict ───────────────────────────────────────────────────────────────────

pub fn predict_impl_multiclass(
    x: ArrayView2<f32>,
    w: ArrayView2<f32>,
) -> ndarray::Array1<usize> {
    let scores = x.dot(&w.t());
    ndarray::Array1::from_iter(scores.outer_iter().map(|row| {
        let mut best_idx = 0;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &v) in row.iter().enumerate() {
            if v > best_val { best_val = v; best_idx = i; }
        }
        best_idx
    }))
}

pub fn predict_proba_impl_multiclass(
    x: ArrayView2<f32>,
    w: ArrayView2<f32>,
) -> Array2<f32> {
    let mut scores = x.dot(&w.t());
    scores.outer_iter_mut().for_each(|mut row| {
        let max = row.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut sum = 0.0f32;
        row.mapv_inplace(|v| { let e = (v - max).exp(); sum += e; e });
        let inv = sum.recip();
        row.mapv_inplace(|v| v * inv);
    });
    scores
}