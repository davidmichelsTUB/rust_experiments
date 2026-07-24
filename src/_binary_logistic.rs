use ndarray::parallel::prelude::*;
use ndarray::s;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use std::mem::MaybeUninit;


/* ORIGINAL per-row parallel implementation — kept for reference / benchmarking.
   Replaced by the gemv version below: for predict_proba the whole thing is a
   single mat-vec X·w, so rayon's per-row dispatch was pure overhead at large n
   (the length-`cols` dot per row is dominated by iterator/closure cost, not flops)
   and produced the high-variance, sometimes-slower-than-sklearn timings.

pub fn dot_sigmoid_fused_parallel(
    x: ArrayView2<f32>,
    w: ArrayView1<f32>,
    bias: f32,
    intercept: bool,
) -> Array2<f32> {
    /* This function is going to be the "predict_proba" and is implemented in the following way:
        - If intercept is true, we will add the bias to the dot product of x and w, and then apply the sigmoid function to get the probability of class 1. The probability of class 0 will be 1 - probability of class 1.
        - If intercept is false, we will just apply the sigmoid function to the dot product of x and w to get the probability of class 1. The probability of class 0 will be 1 - probability of class 1.
     */
    let mut out = Array2::<f32>::uninit((x.nrows(), 2)); // Here we create an uninitialized array of shape (n_samples, 2) to store the probabilities of class 0 and class 1 for each sample.

    match intercept { // this is the bias (intercept) case
        true => {
            out.axis_iter_mut(Axis(0))
                .into_par_iter()
                .zip(x.axis_iter(Axis(0)).into_par_iter())
                .for_each(|(mut out_row, x_row)| {
                    let prob = sigmoid(x_row.dot(&w) + bias); // Here we compute the probability of class 1 by applying the sigmoid function to the dot product of x_row and w, plus the bias.
                    out_row[0] = MaybeUninit::new(1.0 - prob); // Here we compute the probability of class 0 by subtracting the probability of class 1 from 1.
                    out_row[1] = MaybeUninit::new(prob);
                });
        }
        _ => { // this is the no bias (no intercept) case
            out.axis_iter_mut(Axis(0))
                .into_par_iter()
                .zip(x.axis_iter(Axis(0)).into_par_iter())
                .for_each(|(mut out_row, x_row)| {
                    let prob = sigmoid(x_row.dot(&w)); // Here we compute the probability of class 1 by applying the sigmoid function to the dot product of x_row and w.
                    out_row[0] = MaybeUninit::new(1.0 - prob); // Here we compute the probability of class 0 by subtracting the probability of class 1 from 1.
                    out_row[1] = MaybeUninit::new(prob);
                });
        }
    }

    unsafe { out.assume_init() } // Here we assume that the uninitialized array is now fully initialized, since we iterated through it and return it as a regular Array2<f32>.
}
*/

pub fn dot_sigmoid_fused_parallel(
    x: ArrayView2<f32>,
    w: ArrayView1<f32>,
    bias: f32,
    intercept: bool,
) -> Array2<f32> {
    /* "predict_proba", gemv form.
        - Compute every row's score z = w^T · x_row in ONE matrix-vector product
          (x.dot(&w)). This dispatches to ndarray's blocked gemv kernel (BLAS if
          enabled, otherwise matrixmultiply), which is cache-friendly, vectorized,
          and single-pass — the same primitive sklearn uses, hence low variance.
        - Fold the intercept in as a scalar (b) added inside the sigmoid map, so we
          never make a second pass over z just to add the bias.
        - out[:,1] = sigmoid(z + b) = P(class 1); out[:,0] = 1 - that = P(class 0).
     */
    let z = x.dot(&w); // single gemv: score for every sample, shape (n_samples,)
    let b = if intercept { bias } else { 0.0 }; // intercept folded into the map below

    // Fill the (n_samples, 2) output in one serial pass over z. The map is O(n) of
    // cheap scalar work and is dominated by the gemv above, so we keep it serial to
    // avoid re-introducing the rayon dispatch overhead we just removed.
    let mut out = Array2::<f32>::uninit((z.len(), 2));
    out.axis_iter_mut(Axis(0))
        .zip(z.iter())
        .for_each(|(mut out_row, &zi)| {
            let prob = sigmoid(zi + b); // P(class 1)
            out_row[0] = MaybeUninit::new(1.0 - prob); // P(class 0)
            out_row[1] = MaybeUninit::new(prob);
        });

    unsafe { out.assume_init() } // every row was written above
}


/* ORIGINAL per-row parallel implementation — kept for reference / benchmarking.
   Replaced by the gemv version below for the same reason as predict_proba: predict
   is a single mat-vec followed by a sign test, so the per-row parallel dot was
   overhead-bound and lost to sklearn at large n / small cols.

pub fn dot_argmax_binary(
    x: ArrayView2<f32>,
    w: ArrayView1<f32>,
    bias: f32,
    intercept: bool,
) -> Array1<u32> {
    /* This function is going to be the "predict" and is implemented in the following way:
        - If intercept is true, we will simply compute the dot product of x and w, add the bias, and then return 1 if the result is greater than or equal to 0, and 0 otherwise.
        - If intercept is false, we will simply compute the dot product of x and w, and then return 1 if the result is greater than or equal to 0, and 0 otherwise.
        Why does this work?
        - In logistic regression, the decision boundary is defined by the equation w^T * x + b = 0. If the result of this equation is greater than or equal to 0, we predict class 1, otherwise we predict class 0. This is equivalent to taking the argmax of the probabilities of class 0 and class 1, which are computed using the sigmoid function. Since the sigmoid function is monotonically increasing, the argmax will be the same as checking if the result of w^T * x + b is greater than or equal to 0.
     */
    let mut out = Array1::<u32>::uninit(x.nrows()); // Here we create an uninitialized array of shape (n_samples) to store the predicted class for each sample.

    match intercept { // this is the bias (intercept) case
        true => {
            out.as_slice_mut()
                .unwrap()
                .par_iter_mut()
                .zip(x.axis_iter(Axis(0)).into_par_iter())
                .for_each(|(out_i, x_row)| {
                    let z = x_row.dot(&w) + bias; // compute w^T * x + b
                    *out_i = MaybeUninit::new((z >= 0.0) as u32); // assign class label
                });
        }
        _ => { // this is the no bias (no intercept) case
            out.as_slice_mut()
                .unwrap()
                .par_iter_mut()
                .zip(x.axis_iter(Axis(0)).into_par_iter())
                .for_each(|(out_i, x_row)| {
                    let z = x_row.dot(&w); // compute w^T * x
                    *out_i = MaybeUninit::new((z >= 0.0) as u32); // assign class label
                });
        }
    }

    unsafe { out.assume_init() } // Here we assume that the uninitialized array is now fully initialized, since we iterated through it and return it as a regular Array1<u32>.
}
*/

pub fn dot_argmax_binary(
    x: ArrayView2<f32>,
    w: ArrayView1<f32>,
    bias: f32,
    intercept: bool,
) -> Array1<u32> {
    /* "predict", gemv form.
        - Score every sample with a single mat-vec z = x.dot(&w) (blocked, vectorized
          gemv), then fold the intercept in as a scalar and threshold at 0.
        - Because sigmoid is monotonically increasing, argmax over [P(class0), P(class1)] is
          exactly the sign test z + b >= 0 → class 1, so we skip the sigmoid entirely.
        - mapv allocates the u32 output in one O(n) pass; the gemv dominates the cost.
     */
    let z = x.dot(&w); // single gemv: w^T · x_row for every sample
    let b = if intercept { bias } else { 0.0 }; // intercept folded into the threshold

    z.mapv(|zi| (zi + b >= 0.0) as u32) // class 1 iff z + b >= 0, else class 0
}

///

pub fn compute_loss_binary_intercept(
    x: ArrayView2<f32>,
    y: ArrayView1<u32>,
    w: ArrayView1<f32>,
    l1_reg: f32,
    l2_reg: f32,
    sample_weights_sum: f32,
    sample_weights: Option<ArrayView1<f32>>,
) -> f32 {
    let w_len = w.len();
    let (bias, base_w) = (w[w_len - 1], w.slice(s![..w_len - 1]));

    let reg = base_w
        .iter()
        .map(|wi| l1_reg * wi.abs() + 0.5 * l2_reg * wi * wi)
        .sum::<f32>();

    let log_loss = match sample_weights {
        Some(weights) => x
            .axis_iter(Axis(0))
            .into_par_iter()
            .zip(y.as_slice().unwrap().into_par_iter())
            .zip(weights.as_slice().unwrap().into_par_iter())
            .map(|((row, label), weight)| {
                let z = row.dot(&base_w) + bias;
                (softplus(z) - (*label as f32) * z) * *weight
            })
            .sum::<f32>(),
        None => x
            .axis_iter(Axis(0))
            .into_par_iter()
            .zip(y.as_slice().unwrap().into_par_iter())
            .map(|(row, label)| {
                let z = row.dot(&base_w) + bias;
                softplus(z) - (*label as f32) * z
            })
            .sum::<f32>(),
    };

    (log_loss + reg) / sample_weights_sum
}

pub fn compute_loss_binary_no_intercept(
    x: ArrayView2<f32>,
    y: ArrayView1<u32>,
    w: ArrayView1<f32>,
    l1_reg: f32,
    l2_reg: f32,
    sample_weights_sum: f32,
    sample_weights: Option<ArrayView1<f32>>,
) -> f32 {

    let reg = w.iter().map(|wi| l1_reg * wi.abs() + 0.5 * l2_reg * wi * wi).sum::<f32>();

    let log_loss = match sample_weights {
        Some(weights) => x
            .axis_iter(Axis(0))
            .into_par_iter()
            .zip(y.as_slice().unwrap().into_par_iter())
            .zip(weights.as_slice().unwrap().into_par_iter())
            .map(|((row, label), weight)| {
                let z = row.dot(&w);
                (softplus(z) - (*label as f32) * z) * *weight
            })
            .sum::<f32>(),
        None => x
            .axis_iter(Axis(0))
            .into_par_iter()
            .zip(y.as_slice().unwrap().into_par_iter())
            .map(|(row, label)| {
                let z = row.dot(&w);
                softplus(z) - (*label as f32) * z
            })
            .sum::<f32>(),
    };

    (log_loss + reg) / sample_weights_sum
}

///

pub fn compute_new_gradient_binary_with_intercept(
    x: ArrayView2<f32>,
    y: ArrayView1<u32>,
    w: ArrayView1<f32>,
    l1_reg: f32,
    l2_reg: f32,
    sample_weights_sum: f32,
    sample_weights: Option<ArrayView1<f32>>,
) -> Array1<f32> {

    let w_len = w.len();
    let (bias, base_w) = (w[w_len - 1], w.slice(s![..w_len - 1]));


    // We zip x's rows directly with the y (and weights) slices instead of using
    // `.enumerate()` + `y[i]`/`weights[i]`. Indexed access into an ndarray goes
    // through a bounds check on every lookup; zipping over the raw slices hands
    // the iterator matching pairs by position, so each row reads its label/weight
    // with no bounds check and no index arithmetic. `as_slice().unwrap()` is safe
    // because y and weights are contiguous 1-D views.
    let mut loss_grad = match sample_weights {
            Some(weights) => x
                .axis_iter(Axis(0))
                .into_par_iter()
                .zip(y.as_slice().unwrap().into_par_iter())
                .zip(weights.as_slice().unwrap().into_par_iter())
                .fold(
                    || Array1::<f32>::zeros(w_len),
                    |mut acc, ((row, label), weight)| {
                        let z = row.dot(&base_w) + bias;
                        // Hoist the per-row scalar out of the inner update: every
                        // feature is scaled by the same `weight * (sigmoid - y)`,
                        // so we compute it once instead of re-multiplying inside a loop.
                        let coeff = *weight * (sigmoid(z) - (*label as f32));

                        // `acc[..d] += coeff * row` as a single fused axpy. This replaces
                        // the hand-written `for j { acc[j] += ... }` loop, which paid a
                        // bounds check per feature and did not auto-vectorize well.
                        // scaled_add is ndarray's tuned kernel: contiguous, SIMD-friendly,
                        // one pass. The bias term sits at index w_len-1 and is updated
                        // separately since it has no matching x column.
                        acc.slice_mut(s![..w_len - 1]).scaled_add(coeff, &row);
                        acc[w_len - 1] += coeff;

                        acc
                    },
                )
                .reduce(|| Array1::<f32>::zeros(w_len), |a, b| a + b),
            None => x
                .axis_iter(Axis(0))
                .into_par_iter()
                .zip(y.as_slice().unwrap().into_par_iter())
                .fold(
                    || Array1::<f32>::zeros(w_len),
                    |mut acc, (row, label)| {
                        let z = row.dot(&base_w) + bias;
                        let coeff = sigmoid(z) - (*label as f32);

                        acc.slice_mut(s![..w_len - 1]).scaled_add(coeff, &row);
                        acc[w_len - 1] += coeff;

                        acc
                    },
                )
                .reduce(|| Array1::<f32>::zeros(w_len), |a, b| a + b)
    };

    for (g, &wi) in loss_grad.iter_mut().zip(base_w.iter()) {
        *g =  (*g + l1_reg * wi.signum() + l2_reg * wi)/sample_weights_sum;
    }

    loss_grad[w_len - 1] /= sample_weights_sum;

    loss_grad
}

pub fn compute_new_gradient_binary_no_intercept(
    x: ArrayView2<f32>,
    y: ArrayView1<u32>,
    w: ArrayView1<f32>,
    l1_reg: f32,
    l2_reg: f32,
    sample_weights_sum: f32,
    sample_weights: Option<ArrayView1<f32>>,
) -> Array1<f32> {
    // Same idea as the intercept version: zip the y (and weights) slices with x's
    // rows to skip per-lookup bounds checks, hoist the shared per-row scalar into
    // `coeff`, and accumulate with a single fused axpy instead of an indexed loop.
    // No bias column here, so `acc` and `row` are the same length and scaled_add
    // covers the whole vector in one call.
    let mut loss_grad = match sample_weights {
            Some(weights) => x
                .axis_iter(Axis(0))
                .into_par_iter()
                .zip(y.as_slice().unwrap().into_par_iter())
                .zip(weights.as_slice().unwrap().into_par_iter())
                .fold(
                    || Array1::<f32>::zeros(w.len()),
                    |mut acc, ((row, label), weight)| {
                        let z = row.dot(&w);
                        // One scalar per row; every feature shares it.
                        let coeff = *weight * (sigmoid(z) - (*label as f32));

                        // acc += coeff * row, vectorized in one pass.
                        acc.scaled_add(coeff, &row);

                        acc
                    },
                )
                .reduce(|| Array1::<f32>::zeros(w.len()), |a, b| a + b),
            None => x
                .axis_iter(Axis(0))
                .into_par_iter()
                .zip(y.as_slice().unwrap().into_par_iter())
                .fold(
                    || Array1::<f32>::zeros(w.len()),
                    |mut acc, (row, label)| {
                        let z = row.dot(&w);
                        let coeff = sigmoid(z) - (*label as f32);

                        acc.scaled_add(coeff, &row);

                        acc
                    },
                )
                .reduce(|| Array1::<f32>::zeros(w.len()), |a, b| a + b)
    };

    for (g, &wi) in loss_grad.iter_mut().zip(w.iter()) {
        *g =  (*g + l1_reg * wi.signum() + l2_reg * wi)/sample_weights_sum;
    }

    loss_grad
}


// Helper Functions

#[inline(always)]
fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

// sigmoid, evaluated in a numerically stable way:
//   max(z, 0) + log(1 + exp(-|z|))
// Never overflows/underflows for finite z, and ln_1p keeps precision
// for small arguments. Used for the loss instead of routing through
// sigmoid -> ln, which produces -inf once sigmoid saturates to 0/1.
#[inline(always)]
fn softplus(z: f32) -> f32 {
    z.max(0.0) + (-z.abs()).exp().ln_1p()
}
