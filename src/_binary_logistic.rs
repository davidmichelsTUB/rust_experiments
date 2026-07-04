use ndarray::parallel::prelude::*;
use ndarray::s;
use ndarray::{Array2, Array1, ArrayView1, ArrayView2, Axis};
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
pub fn dot_sigmoid_fused_parallel(x: ArrayView2<f32>, w: ArrayView1<f32>, intercept: bool) -> Array2<f32> {
    
    let w_len = w.len();
    
    let (bias, base_w) = (w[w_len - 1], w.slice(s![..]));

    let (bias, w_actual) = if intercept {
        (bias, base_w.slice(s![..w_len - 1]))
    } else {
        (0.0, base_w)
    };
    
    let mut out = Array2::<f32>::uninit((x.nrows(), 2));

    out.axis_iter_mut(Axis(0))
        .into_par_iter()
        .zip(x.axis_iter(Axis(0)).into_par_iter())
        .for_each(|(mut out_row, x_row)| {
            let prob = sigmoid(x_row.dot(&w_actual) + bias);
            out_row[0] = MaybeUninit::new(prob);
            out_row[1] = MaybeUninit::new(1.0 - prob);
        });
        
    unsafe {out.assume_init()}
}

pub fn dot_argmax_binary(x: ArrayView2<f32>, w: ArrayView1<f32>, intercept: bool) -> Array1<u32> {
    let w_len = w.len();
    
    let (bias, base_w) = (w[w_len - 1], w.slice(s![..]));

    let (bias, w_actual) = if intercept {
        (bias, base_w.slice(s![..w_len - 1]))
    } else {
        (0.0, base_w)
    };
    
    let mut out = Array1::<u32>::uninit(x.nrows());

    out.as_slice_mut().unwrap()
    .par_iter_mut()
    .zip(x.axis_iter(Axis(0)).into_par_iter())
    .for_each(|(out_i, x_row)| {
        let z = x_row.dot(&w_actual) + bias;
        *out_i = MaybeUninit::new((z >= 0.0) as u32);
    });
    unsafe {out.assume_init()}
}

// Here the "real" function will call the kernel
// pub fn predict_proba_impl_binary(
//     x: ArrayView2<f32>,
//     w: ArrayView1<f32>,
//     intercept: bool,
//     kernel: &str,
// ) -> Array2<f32> {
//     match kernel {
//         "fused_parallel" => dot_sigmoid_fused_parallel(x, w, intercept),
//         "blas" => dot_sigmoid_blas(x, w),
//         _ => panic!("Unknown kernel"),
//     }
// }

////////////////////////////////////////////////////////////////////////

// Loss has its own kernel
pub fn compute_loss_binary(
    x: ArrayView2<f32>,
    y: ArrayView1<u32>,
    w: ArrayView1<f32>,
    intercept: bool,
    l1_reg: f32,
    l2_reg: f32,
    sample_weights_sum: f32,
    sample_weights: Option<ArrayView1<f32>>,
) -> f32 {
    use ndarray::s;

    let w_len = w.len();
    let (bias, base_w) = (w[w_len - 1], w.slice(s![..]));

    let (bias, w_actual) = if intercept {
        (bias, base_w.slice(s![..w_len - 1]))
    } else {
        (0.0, base_w)
    };

    let (log_loss, reg) = rayon::join(
        || match sample_weights {
            Some(weights) => x
                .axis_iter(Axis(0))
                .into_par_iter()
                .zip(y.as_slice().unwrap().into_par_iter())
                .zip(weights.as_slice().unwrap().into_par_iter())
                .map(|((row, label), weight)| {
                    let z = row.dot(&w_actual) + bias;
                    (softplus(z) - (*label as f32) * z) * *weight
                })
                .sum::<f32>(),
            None => x
                .axis_iter(Axis(0))
                .into_par_iter()
                .zip(y.as_slice().unwrap().into_par_iter())
                .map(|(row, label)| {
                    let z = row.dot(&w_actual) + bias;
                    softplus(z) - (*label as f32) * z
                })
                .sum::<f32>(),
        },
        || {
            w.iter()
                .map(|wi| l1_reg * wi.abs() + 0.5 * l2_reg * wi * wi)
                .sum::<f32>()
        },
    );
    (log_loss + reg) / sample_weights_sum
}

// Gradient has its own kernel
pub fn compute_new_gradient_binary(
    x: ArrayView2<f32>,
    y: ArrayView1<u32>,
    w: ArrayView1<f32>,
    intercept: bool,
    l1_reg: f32,
    l2_reg: f32,
    sample_weights_sum: f32,
    sample_weights: Option<ArrayView1<f32>>,
) -> Array1<f32> {

    let w_len = w.len();
    let (bias, base_w) = (w[w_len - 1], w.slice(s![..]));

    let (bias, w_actual) = if intercept {
        (bias, base_w.slice(s![..w_len - 1]))
    } else {
        (0.0, base_w)
    };

    let (loss_grad, reg_grad) = rayon::join(
        || match sample_weights {
            Some(weights) => x
                .axis_iter(Axis(0))
                .into_par_iter()
                .enumerate()
                .fold(
                    || Array1::<f32>::zeros(w_len),
                    |mut acc, (i, row)| {
                        let z = row.dot(&w_actual) + bias;
                        let diff = sigmoid(z) - (y[i] as f32);
                        let weight = weights[i];

                        for (j, xi) in row.iter().enumerate() {
                            acc[j] += weight * xi * diff;
                        }

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
                        let z = row.dot(&w_actual) + bias;
                        let diff = sigmoid(z) - (y[i] as f32);

                        for (j, xi) in row.iter().enumerate() {
                            acc[j] += xi * diff;
                        }

                        acc
                    },
                )
                .reduce(|| Array1::<f32>::zeros(w_len), |a, b| a + b),
        },
        || {
            w.iter()
                .map(|&wi| l1_reg * wi.signum() + l2_reg * wi)
                .collect::<Array1<f32>>()
        },
    );

    (loss_grad + reg_grad) / sample_weights_sum
}
