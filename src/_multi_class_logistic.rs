use ndarray::parallel::prelude::*;
use ndarray::s;
use ndarray::{Array1, Array2, ArrayBase, ArrayView1, ArrayView2, Axis, Data, Ix1, Ix2};
use std::mem::MaybeUninit;

#[inline]
fn fill_multiclass_logits<SX, SW>(
    x_row: &ArrayBase<SX, Ix1>,
    w_base: &ArrayBase<SW, Ix2>,
    bias: Option<ArrayView1<f32>>,
    logits: &mut [f32],
) where
    SX: Data<Elem = f32>,
    SW: Data<Elem = f32>,
{
    logits.fill(0.0);

    for (feature_idx, &x_value) in x_row.iter().enumerate() {
        let w_row = w_base.row(feature_idx);

        for (logit, &weight) in logits.iter_mut().zip(w_row.iter()) {
            *logit += x_value * weight;
        }
    }

    if let Some(bias) = bias {
        for (logit, &bias_value) in logits.iter_mut().zip(bias.iter()) {
            *logit += bias_value;
        }
    }
}

#[inline]
fn softmax_in_place(row: &mut [f32]) {
    let mut max = f32::NEG_INFINITY;

    for &value in row.iter() {
        if value > max {
            max = value;
        }
    }

    let mut sum = 0.0;

    for value in row.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }

    let inv_sum = 1.0 / sum;

    for value in row.iter_mut() {
        *value *= inv_sum;
    }
}

#[inline]
fn row_log_loss(row: &[f32], label: usize) -> f32 {
    let mut max = f32::NEG_INFINITY;

    for &value in row.iter() {
        if value > max {
            max = value;
        }
    }

    let mut sum = 0.0;

    for &value in row.iter() {
        sum += (value - max).exp();
    }

    max + sum.ln() - row[label]
}

pub fn dot_argmax_multiclass_nointercept(x: ArrayView2<f32>, w: ArrayView2<f32>) -> Array1<u32> {
    let mut out = Array1::<u32>::uninit(x.shape()[0]);

    let result = x.dot(&w);

    out.as_slice_mut()
        .unwrap()
        .par_iter_mut()
        .zip(result.axis_iter(Axis(0)).into_par_iter())
        .for_each(|(out_i, z)| {
            let mut best_idx = 0;
            let mut best_val = z[0];

            for (i, &v) in z.iter().enumerate().skip(1) {
                if v > best_val {
                    best_val = v;
                    best_idx = i as u32;
                }
            }

            *out_i = MaybeUninit::new(best_idx);
        });
    


    unsafe { out.assume_init() }
}

pub fn dot_argmax_multiclass_intercept(x: ArrayView2<f32>, w: ArrayView2<f32>, bias: ArrayView1<f32>) -> Array1<u32> {
    let mut out = Array1::<u32>::uninit(x.shape()[0]);

    let mut result = x.dot(&w);

    out.as_slice_mut()
        .unwrap()
        .par_iter_mut()
        .zip(result.axis_iter_mut(Axis(0)).into_par_iter())
        .for_each(|(out_i, mut z)| {

            z += &bias; 

            let mut best_idx = 0;
            let mut best_val = z[0];

            for (i, &v) in z.iter().enumerate().skip(1) {
                if v > best_val {
                    best_val = v;
                    best_idx = i as u32;
                }
            }

            *out_i = MaybeUninit::new(best_idx);
        });
    


    unsafe { out.assume_init() }
}

pub fn dot_softmax_multiclass_nointercept(x: ArrayView2<f32>, w: ArrayView2<f32>) -> Array2<f32> {
    let mut score = x.dot(&w);
    

    score
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .for_each(|mut row| {
            let mut max = f32::NEG_INFINITY;

            for &v in row.iter() {
                if v > max {
                    max = v;
                }
            }

            let mut sum: f32 = 0.0;

            for v in row.iter_mut() {
                *v = (*v - max).exp();
                sum += *v;
            }

            let inv_sum = 1.0 / sum;

            for v in row.iter_mut() {
                *v *= inv_sum;
            }
        });

    score
}

pub fn dot_softmax_multiclass_intercept(
    x: ArrayView2<f32>,
    w: ArrayView2<f32>,
    bias: ArrayView1<f32>,
) -> Array2<f32> {
    let mut score = x.dot(&w);

    score
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .for_each(|mut row| {
            for (v, &b) in row.iter_mut().zip(bias.iter()) {
                *v += b;
            }

            let mut max = f32::NEG_INFINITY;

            for &v in row.iter() {
                if v > max {
                    max = v;
                }
            }

            let mut sum: f32 = 0.0;

            for v in row.iter_mut() {
                *v = (*v - max).exp();
                sum += *v;
            }

            let inv_sum = 1.0 / sum;

            for v in row.iter_mut() {
                *v *= inv_sum;
            }
        });

    score
}

pub fn compute_loss_multiclass_intercept(
    x: ArrayView2<f32>,
    y: ArrayView1<u32>,
    w: ArrayView1<f32>,
    n_classes: u32,
    n_features: u32,
    l1_reg: f32,
    l2_reg: f32,
    inv_sample_weights_sum: f32,
    sample_weights: Option<ArrayView1<f32>>,
) -> f32 {
    let (bias, w_base) = (
        w.slice(s![..n_classes as usize]),
        w.slice(s![n_classes as usize..]),
    );

    let reg = w_base
        .iter()
        .map(|&wi| l1_reg * wi.abs() + 0.5 * l2_reg * wi * wi)
        .sum::<f32>();

    let w_base = w_base
        .to_shape((n_features as usize, n_classes as usize))
        .unwrap();

    let log_loss = match sample_weights {
        Some(weights) => x
            .axis_iter(Axis(0))
            .into_par_iter()
            .zip(y.as_slice().unwrap())
            .zip(weights.as_slice().unwrap())
            .fold(
                || (0.0f32, vec![0.0f32; n_classes as usize]),
                |(acc, mut logits), ((x_row, &label), &sw)| {
                    fill_multiclass_logits(&x_row, &w_base, Some(bias), &mut logits);
                    let row_loss = row_log_loss(&logits, label as usize) * sw;
                    (acc + row_loss, logits)
                },
            )
            .map(|(acc, _)| acc)
            .sum::<f32>(),
        None => x
            .axis_iter(Axis(0))
            .into_par_iter()
            .zip(y.as_slice().unwrap())
            .fold(
                || (0.0f32, vec![0.0f32; n_classes as usize]),
                |(acc, mut logits), (x_row, &label)| {
                    fill_multiclass_logits(&x_row, &w_base, Some(bias), &mut logits);
                    let row_loss = row_log_loss(&logits, label as usize);
                    (acc + row_loss, logits)
                },
            )
            .map(|(acc, _)| acc)
            .sum::<f32>(),
    };

    (log_loss + reg) * inv_sample_weights_sum
}

pub fn compute_loss_multiclass_nointercept(
    x: ArrayView2<f32>,
    y: ArrayView1<u32>,
    w: ArrayView1<f32>,
    n_classes: u32,
    n_features: u32,
    l1_reg: f32,
    l2_reg: f32,
    inv_sample_weights_sum: f32,
    sample_weights: Option<ArrayView1<f32>>,
) -> f32 {
    let w = w
        .to_shape((n_features as usize, n_classes as usize))
        .unwrap();

    let reg = w
        .iter()
        .map(|&wi| l1_reg * wi.abs() + 0.5 * l2_reg * wi * wi)
        .sum::<f32>();

    let log_loss = match sample_weights {
        Some(weights) => x
            .axis_iter(Axis(0))
            .into_par_iter()
            .zip(y.as_slice().unwrap())
            .zip(weights.as_slice().unwrap())
            .fold(
                || (0.0f32, vec![0.0f32; n_classes as usize]),
                |(acc, mut logits), ((x_row, &label), &sw)| {
                    fill_multiclass_logits(&x_row, &w, None, &mut logits);
                    let row_loss = row_log_loss(&logits, label as usize) * sw;
                    (acc + row_loss, logits)
                },
            )
            .map(|(acc, _)| acc)
            .sum::<f32>(),
        None => x
            .axis_iter(Axis(0))
            .into_par_iter()
            .zip(y.as_slice().unwrap())
            .fold(
                || (0.0f32, vec![0.0f32; n_classes as usize]),
                |(acc, mut logits), (x_row, &label)| {
                    fill_multiclass_logits(&x_row, &w, None, &mut logits);
                    let row_loss = row_log_loss(&logits, label as usize);
                    (acc + row_loss, logits)
                },
            )
            .map(|(acc, _)| acc)
            .sum::<f32>(),
    };

    (log_loss + reg) * inv_sample_weights_sum
}

pub fn compute_gradient_multiclass_nointercept(
    x: ArrayView2<f32>,
    _x_t: ArrayView2<f32>,
    y: ArrayView1<u32>,
    w: ArrayView1<f32>,
    n_classes: u32,
    n_features: u32,
    l1_reg: f32,
    l2_reg: f32,
    inv_sample_weights_sum: f32,
    sample_weights: Option<ArrayView1<f32>>,
) -> Array1<f32> {
    let n_classes = n_classes as usize;
    let n_features = n_features as usize;
    let weights_len = n_features * n_classes;
    let w = w.to_shape((n_features, n_classes)).unwrap();

    let (_, mut weight_grad) = match sample_weights {
        Some(weights) => x
            .axis_iter(Axis(0))
            .into_par_iter()
            .zip(y.as_slice().unwrap())
            .zip(weights.as_slice().unwrap())
            .fold(
                || (vec![0.0f32; weights_len], vec![0.0f32; n_classes]),
                |(mut weight_grad, mut logits), ((x_row, &label), &sw)| {
                    fill_multiclass_logits(&x_row, &w, None, &mut logits);
                    softmax_in_place(&mut logits);
                    logits[label as usize] -= 1.0;

                    for value in logits.iter_mut() {
                        *value *= sw;
                    }

                    for (feature_idx, &x_value) in x_row.iter().enumerate() {
                        let base = feature_idx * n_classes;

                        for class_idx in 0..n_classes {
                            weight_grad[base + class_idx] += x_value * logits[class_idx];
                        }
                    }

                    (weight_grad, logits)
                },
            )
            .reduce(
                || (vec![0.0f32; weights_len], vec![0.0f32; n_classes]),
                |mut left, right| {
                    for (dst, src) in left.0.iter_mut().zip(right.0.iter()) {
                        *dst += *src;
                    }
                    left
                },
            ),
        None => x
            .axis_iter(Axis(0))
            .into_par_iter()
            .zip(y.as_slice().unwrap())
            .fold(
                || (vec![0.0f32; weights_len], vec![0.0f32; n_classes]),
                |(mut weight_grad, mut logits), (x_row, &label)| {
                    fill_multiclass_logits(&x_row, &w, None, &mut logits);
                    softmax_in_place(&mut logits);
                    logits[label as usize] -= 1.0;

                    for (feature_idx, &x_value) in x_row.iter().enumerate() {
                        let base = feature_idx * n_classes;

                        for class_idx in 0..n_classes {
                            weight_grad[base + class_idx] += x_value * logits[class_idx];
                        }
                    }

                    (weight_grad, logits)
                },
            )
            .reduce(
                || (vec![0.0f32; weights_len], vec![0.0f32; n_classes]),
                |mut left, right| {
                    for (dst, src) in left.0.iter_mut().zip(right.0.iter()) {
                        *dst += *src;
                    }
                    left
                },
            ),
    };

    for (g, &wi) in weight_grad.iter_mut().zip(w.iter()) {
        *g += l1_reg * wi.signum() + l2_reg * wi;
    }

    for g in weight_grad.iter_mut() {
        *g *= inv_sample_weights_sum;
    }

    Array1::from_vec(weight_grad)
}

pub fn compute_gradient_multiclass_intercept(
    x: ArrayView2<f32>,
    _x_t: ArrayView2<f32>,
    y: ArrayView1<u32>,
    w: ArrayView1<f32>,
    n_classes: u32,
    n_features: u32,
    l1_reg: f32,
    l2_reg: f32,
    inv_sample_weights_sum: f32,
    sample_weights: Option<ArrayView1<f32>>,
) -> Array1<f32> {
    let n_classes = n_classes as usize;
    let n_features = n_features as usize;
    let weights_len = n_features * n_classes;

    let (bias, w_base) = (w.slice(s![..n_classes]), w.slice(s![n_classes..]));

    let w_base = w_base.to_shape((n_features, n_classes)).unwrap();

    let (bias_grad, mut weight_grad, _) = match sample_weights {
        Some(weights) => x
            .axis_iter(Axis(0))
            .into_par_iter()
            .zip(y.as_slice().unwrap())
            .zip(weights.as_slice().unwrap())
            .fold(
                || {
                    (
                        vec![0.0f32; n_classes],
                        vec![0.0f32; weights_len],
                        vec![0.0f32; n_classes],
                    )
                },
                |(mut bias_grad, mut weight_grad, mut logits), ((x_row, &label), &sw)| {
                    fill_multiclass_logits(&x_row, &w_base, Some(bias), &mut logits);
                    softmax_in_place(&mut logits);
                    logits[label as usize] -= 1.0;

                    for value in logits.iter_mut() {
                        *value *= sw;
                    }

                    for (dst, &value) in bias_grad.iter_mut().zip(logits.iter()) {
                        *dst += value;
                    }

                    for (feature_idx, &x_value) in x_row.iter().enumerate() {
                        let base = feature_idx * n_classes;

                        for class_idx in 0..n_classes {
                            weight_grad[base + class_idx] += x_value * logits[class_idx];
                        }
                    }

                    (bias_grad, weight_grad, logits)
                },
            )
            .reduce(
                || {
                    (
                        vec![0.0f32; n_classes],
                        vec![0.0f32; weights_len],
                        vec![0.0f32; n_classes],
                    )
                },
                |mut left, right| {
                    for (dst, src) in left.0.iter_mut().zip(right.0.iter()) {
                        *dst += *src;
                    }
                    for (dst, src) in left.1.iter_mut().zip(right.1.iter()) {
                        *dst += *src;
                    }
                    left
                },
            ),
        None => x
            .axis_iter(Axis(0))
            .into_par_iter()
            .zip(y.as_slice().unwrap())
            .fold(
                || {
                    (
                        vec![0.0f32; n_classes],
                        vec![0.0f32; weights_len],
                        vec![0.0f32; n_classes],
                    )
                },
                |(mut bias_grad, mut weight_grad, mut logits), (x_row, &label)| {
                    fill_multiclass_logits(&x_row, &w_base, Some(bias), &mut logits);
                    softmax_in_place(&mut logits);
                    logits[label as usize] -= 1.0;

                    for (dst, &value) in bias_grad.iter_mut().zip(logits.iter()) {
                        *dst += value;
                    }

                    for (feature_idx, &x_value) in x_row.iter().enumerate() {
                        let base = feature_idx * n_classes;

                        for class_idx in 0..n_classes {
                            weight_grad[base + class_idx] += x_value * logits[class_idx];
                        }
                    }

                    (bias_grad, weight_grad, logits)
                },
            )
            .reduce(
                || {
                    (
                        vec![0.0f32; n_classes],
                        vec![0.0f32; weights_len],
                        vec![0.0f32; n_classes],
                    )
                },
                |mut left, right| {
                    for (dst, src) in left.0.iter_mut().zip(right.0.iter()) {
                        *dst += *src;
                    }
                    for (dst, src) in left.1.iter_mut().zip(right.1.iter()) {
                        *dst += *src;
                    }
                    left
                },
            ),
    };

    for (g, &wi) in weight_grad.iter_mut().zip(w_base.iter()) {
        *g += l1_reg * wi.signum() + l2_reg * wi;
    }

    let mut grad = Vec::with_capacity(n_classes + weights_len);
    grad.extend_from_slice(&bias_grad);
    grad.extend_from_slice(&weight_grad);

    for g in grad.iter_mut() {
        *g *= inv_sample_weights_sum;
    }

    Array1::from_vec(grad)
}
