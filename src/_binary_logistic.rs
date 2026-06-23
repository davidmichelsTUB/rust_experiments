use ndarray::parallel::prelude::*;
use ndarray::{Array, Array1, ArrayView1, ArrayView2, Axis};

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

// Variant 1: Use BLAS dgemv + mapv (two passes)
fn dot_sigmoid_blas(x: ArrayView2<f32>, w: ArrayView1<f32>) -> Array1<f32> {
    x.dot(&w).mapv_into(sigmoid)
}

// Variant 2: Parallelize the dot product + sigmoid using rayon
fn dot_sigmoid_fused_parallel(x: ArrayView2<f32>, w: ArrayView1<f32>) -> Array1<f32> {
    Array::from_vec(
        x.axis_iter(Axis(0))
            .into_par_iter()
            .map(|row| sigmoid(row.dot(&w)))
            .collect(),
    )
}

// Here the "real" function will call the kernel
pub fn predict_proba_impl(x: ArrayView2<f32>, w: ArrayView1<f32>, kernel: &str) -> Array1<f32> {
    match kernel {
        "fused_parallel" => dot_sigmoid_fused_parallel(x, w),
        "blas" => dot_sigmoid_blas(x, w),
        _ => panic!("Unknown kernel"),
    }
}

////////////////////////////////////////////////////////////////////////

// Loss has its own kernel
pub fn compute_loss(
    x: ArrayView2<f32>,
    y: ArrayView1<f32>,
    w: ArrayView1<f32>,
    l1_reg: f32,
    l2_reg: f32,
    sample_weights: Option<ArrayView1<f32>>,
) -> f32 {
    let (log_loss, reg) =  rayon::join(
            || {
                match sample_weights {
                    Some(weights) => {
                x.axis_iter(Axis(0))
                    .into_par_iter()
                    .zip(y.as_slice().unwrap().into_par_iter())
                    .zip(weights.as_slice().unwrap().into_par_iter())
                    .map(|((row, label), weight)| {
                        let z = row.dot(&w);
                        (softplus(z) - *label * z) * *weight
                    })
                    .sum::<f32>()
            }
            None => {
                x.axis_iter(Axis(0))
                    .into_par_iter()
                    .zip(y.as_slice().unwrap().into_par_iter())
                    .map(|(row, label)| {
                        let z = row.dot(&w);
                        softplus(z) - *label * z
                    })
                    .sum::<f32>()
            }
                }
            },
            || {
                w.as_slice()
                    .unwrap()
                    .iter()
                    .map(|wi| l1_reg * wi.abs() + l2_reg * wi * wi)
                    .sum::<f32>()
            },
        );
    (log_loss + reg) / (x.nrows() as f32)
}

// Gradient has its own kernel
pub fn compute_new_gradient(
    x: ArrayView2<f32>,
    y: ArrayView1<f32>,
    w: ArrayView1<f32>,
    l1_reg: f32,
    l2_reg: f32,
    sample_weights: Option<ArrayView1<f32>>,
) -> Array1<f32> {
    let (loss_grad, reg_grad) = rayon::join(
        || {
            match sample_weights {
                Some(weights) => {
                    x.axis_iter(Axis(0))
                        .into_par_iter()
                        .enumerate()
                        .fold(
                            || Array1::<f32>::zeros(w.len()),
                            |mut acc, (i, row)| {
                                let z = row.dot(&w);
                                let diff = sigmoid(z) - y[i];
                                let weight = weights[i];

                                for (j, xi) in row.iter().enumerate() {
                                    acc[j] += weight * xi * diff;
                                }

                                acc
                            },
                        )
                        .reduce(
                            || Array1::<f32>::zeros(w.len()),
                            |a, b| a + b,
                        )
                }
                None => {
                    x.axis_iter(Axis(0))
                        .into_par_iter()
                        .enumerate()
                        .fold(
                            || Array1::<f32>::zeros(w.len()),
                            |mut acc, (i, row)| {
                                let z = row.dot(&w);
                                let diff = sigmoid(z) - y[i];

                                for (j, xi) in row.iter().enumerate() {
                                    acc[j] += xi * diff;
                                }

                                acc
                            },
                        )
                        .reduce(
                            || Array1::<f32>::zeros(w.len()),
                            |a, b| a + b,
                        )
                }
            }
        },
        || {
            w.iter()
                .map(|&wi| l1_reg * wi.signum() + 2.0 * l2_reg * wi)
                .collect::<Array1<f32>>()
        },
    );

    (loss_grad + reg_grad) / x.nrows() as f32
}