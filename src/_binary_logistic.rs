use ndarray::parallel::prelude::*;
use ndarray::{Array, Array1, ArrayView1, ArrayView2, Axis};

#[inline(always)]
fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

//copy paste from scikit learn
#[inline(always)]
fn log1pexp(z: f32) -> f32 {
    match z {
        z if z <= -17.0 => z.exp(),
        z if z <= -1.0 => z.exp().ln_1p(),
        z if z <= 9.0 => (1.0 + z.exp()).ln(),
        z if z <= 14.6 => z + (-z).exp(),
        _ => z,
    }
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
    sample_weights_sum: f32,
    sample_weights: Option<ArrayView1<f32>>
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
                        (log1pexp(z) - *label * z) * *weight
                    })
                    .sum::<f32>()
            }
            None => {
                x.axis_iter(Axis(0))
                    .into_par_iter()
                    .zip(y.as_slice().unwrap().into_par_iter())
                    .map(|(row, label)| {
                        let z = row.dot(&w);
                        log1pexp(z) - *label * z
                    })
                    .sum::<f32>()
            }
                }
            },
            || {
                w.as_slice()
                    .unwrap()
                    .iter()
                    .map(|wi| l1_reg * wi.abs() + l2_reg * wi * wi * 0.5)
                    .sum::<f32>()
            },
        );

    log_loss / sample_weights_sum + reg
}

// Gradient has its own kernel
pub fn compute_new_gradient(
    x: ArrayView2<f32>,
    y: ArrayView1<f32>,
    w: ArrayView1<f32>,
    l1_reg: f32,
    l2_reg: f32,
    sample_weights_sum: f32,
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
                                let r = row.dot(&w);
                                let z = r.exp();
                                let z_neg = 1.0 / z;

                                let diff = match r {
                                    r if r > -17.0 => ((1.0 - y[i]) - y[i] * z_neg)/(1.0 + z_neg),
                                    _ => z - y[i]
                                };
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
                                let r = row.dot(&w);
                                let z = r.exp();
                                let z_neg = 1.0 / z;

                                let diff = match r {
                                    r if r > -17.0 => ((1.0 - y[i]) - y[i] * z_neg)/(1.0 + z_neg),
                                    _ => z - y[i]
                                };
                                
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
                .map(|&wi| l1_reg * wi.signum() + l2_reg * wi)
                .collect::<Array1<f32>>()
        },
    );

    loss_grad / sample_weights_sum + reg_grad
}