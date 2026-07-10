use ndarray::parallel::prelude::*;
use ndarray::s;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use std::mem::MaybeUninit;

#[inline(always)]
fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

// log(1 + exp(z)), evaluated in a numerically stable way:
//   max(z, 0) + log(1 + exp(-|z|))
// Never overflows/underflows for finite z, and ln_1p keeps precision
// for small arguments. Used for the loss instead of routing through
// sigmoid -> ln, which produces -inf once sigmoid saturates to 0/1.
#[inline(always)]
fn softplus(z: f32) -> f32 {
    z.max(0.0) + (-z.abs()).exp().ln_1p()
}

// // Variant 1: Use BLAS dgemv + mapv (two passes)
// fn dot_sigmoid_blas(x: ArrayView2<f32>, w: ArrayView1<f32>) -> Array1<f32> {
//     let mut result = x.dot(&w);
//     result.mapv_inplace(sigmoid);
//     result
// }

// Variant 2: Parallelize the dot product + sigmoid using rayon
pub fn dot_sigmoid_fused_parallel(
    x: ArrayView2<f32>,
    w: ArrayView1<f32>,
    bias: f32,
    intercept: bool,
) -> Array2<f32> {
    let mut out = Array2::<f32>::uninit((x.nrows(), 2));

    match intercept {
        true => {
            out.axis_iter_mut(Axis(0))
                .into_par_iter()
                .zip(x.axis_iter(Axis(0)).into_par_iter())
                .for_each(|(mut out_row, x_row)| {
                    let prob = sigmoid(x_row.dot(&w) + bias);
                    out_row[0] = MaybeUninit::new(prob);
                    out_row[1] = MaybeUninit::new(1.0 - prob);
                });
        }
        _ => {
            out.axis_iter_mut(Axis(0))
                .into_par_iter()
                .zip(x.axis_iter(Axis(0)).into_par_iter())
                .for_each(|(mut out_row, x_row)| {
                    let prob = sigmoid(x_row.dot(&w));
                    out_row[0] = MaybeUninit::new(prob);
                    out_row[1] = MaybeUninit::new(1.0 - prob);
                });
        }
    }

    unsafe { out.assume_init() }
}

pub fn dot_argmax_binary(
    x: ArrayView2<f32>,
    w: ArrayView1<f32>,
    bias: f32,
    intercept: bool,
) -> Array1<u32> {
    let mut out = Array1::<u32>::uninit(x.nrows());

    match intercept {
        true => {
            out.as_slice_mut()
                .unwrap()
                .par_iter_mut()
                .zip(x.axis_iter(Axis(0)).into_par_iter())
                .for_each(|(out_i, x_row)| {
                    let z = x_row.dot(&w) + bias;
                    *out_i = MaybeUninit::new((z >= 0.0) as u32);
                });
        }
        _ => {
            out.as_slice_mut()
                .unwrap()
                .par_iter_mut()
                .zip(x.axis_iter(Axis(0)).into_par_iter())
                .for_each(|(out_i, x_row)| {
                    let z = x_row.dot(&w);
                    *out_i = MaybeUninit::new((z >= 0.0) as u32);
                });
        }
    }

    unsafe { out.assume_init() }
}

// Loss has its own kernel


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

// Gradient has its own kernel


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


    let mut loss_grad = match sample_weights {
            Some(weights) => x
                .axis_iter(Axis(0))
                .into_par_iter()
                .enumerate()
                .fold(
                    || Array1::<f32>::zeros(w_len),
                    |mut acc, (i, row)| {
                        let z = row.dot(&base_w) + bias;
                        let diff = sigmoid(z) - (y[i] as f32);
                        let weight = weights[i];

                        for (j, xi) in row.iter().enumerate() {
                            acc[j] += weight * xi * diff;
                        }

                        acc[w_len - 1] += weight * diff;

                        acc
                    },
                )
                .reduce(|| Array1::<f32>::zeros(w_len), |a, b| a + b),
            None => x
                .axis_iter(Axis(0))
                .into_par_iter()
                .enumerate()
                .fold(
                    || Array1::<f32>::zeros(w_len),
                    |mut acc, (i, row)| {
                        let z = row.dot(&base_w) + bias;
                        let diff = sigmoid(z) - (y[i] as f32);

                        for (j, xi) in row.iter().enumerate() {
                            acc[j] += xi * diff;
                        }

                        acc[w_len - 1] += diff;

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
    let mut loss_grad = match sample_weights {
            Some(weights) => x
                .axis_iter(Axis(0))
                .into_par_iter()
                .enumerate()
                .fold(
                    || Array1::<f32>::zeros(w.len()),
                    |mut acc, (i, row)| {
                        let z = row.dot(&w);
                        let diff = sigmoid(z) - (y[i] as f32);
                        let weight = weights[i];

                        for (j, xi) in row.iter().enumerate() {
                            acc[j] += weight * xi * diff;
                        }

                        acc
                    },
                )
                .reduce(|| Array1::<f32>::zeros(w.len()), |a, b| a + b),
            None => x
                .axis_iter(Axis(0))
                .into_par_iter()
                .enumerate()
                .fold(
                    || Array1::<f32>::zeros(w.len()),
                    |mut acc, (i, row)| {
                        let z = row.dot(&w);
                        let diff = sigmoid(z) - (y[i] as f32);

                        for (j, xi) in row.iter().enumerate() {
                            acc[j] += xi * diff;
                        }

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