use ndarray::parallel::prelude::*;
use ndarray::s;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use std::mem::MaybeUninit;

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

pub fn dot_argmax_multiclass_intercept(
    x: ArrayView2<f32>,
    w: ArrayView2<f32>,
    bias: ArrayView1<f32>,
) -> Array1<u32> {
    let mut out = Array1::<u32>::uninit(x.nrows());

    let mut result = x.dot(&w);
    result += &bias;

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

    let mut score = x.dot(
        &w_base
            .to_shape((n_features as usize, n_classes as usize))
            .unwrap(),
    );
    score += &bias;

    let log_loss = match sample_weights {
        Some(weights) => score
            .axis_iter(Axis(0))
            .into_par_iter()
            .zip(y.as_slice().unwrap())
            .zip(weights.as_slice().unwrap())
            .map(|((row, &label), &sw)| {
                let mut max = f32::NEG_INFINITY;

                for &v in row {
                    if v > max {
                        max = v;
                    }
                }

                let mut sum: f32 = 0.0;
                for &v in row {
                    sum += (v - max).exp();
                }

                let log_sum_exp = max + sum.ln();

                (log_sum_exp - row[label as usize]) * sw
            })
            .sum::<f32>(),
        None => score
            .axis_iter(Axis(0))
            .into_par_iter()
            .zip(y.as_slice().unwrap())
            .map(|(row, &label)| {
                let mut max = f32::NEG_INFINITY;
                for &v in row {
                    if v > max {
                        max = v;
                    }
                }

                let mut sum: f32 = 0.0;
                for &v in row {
                    sum += (v - max).exp();
                }

                let log_sum_exp = max + sum.ln();

                log_sum_exp - row[label as usize]
            })
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
    let scores = x.dot(
        &w.to_shape((n_features as usize, n_classes as usize))
            .unwrap(),
    );

    let reg = w
        .iter()
        .map(|&wi| l1_reg * wi.abs() + 0.5 * l2_reg * wi * wi)
        .sum::<f32>();

    let log_loss = match sample_weights {
        Some(weights) => scores
            .axis_iter(Axis(0))
            .into_par_iter()
            .zip(y.as_slice().unwrap())
            .zip(weights.as_slice().unwrap())
            .map(|((row, &label), &sw)| {
                let mut max = f32::NEG_INFINITY;
                for &v in row {
                    if v > max {
                        max = v;
                    }
                }

                let mut sum: f32 = 0.0;
                for &v in row {
                    sum += (v - max).exp();
                }

                let log_sum_exp = max + sum.ln();

                (log_sum_exp - row[label as usize]) * sw
            })
            .sum::<f32>(),
        None => scores
            .axis_iter(Axis(0))
            .into_par_iter()
            .zip(y.as_slice().unwrap())
            .map(|(row, &label)| {
                let mut max = f32::NEG_INFINITY;
                for &v in row {
                    if v > max {
                        max = v;
                    }
                }

                let mut sum: f32 = 0.0;
                for &v in row {
                    sum += (v - max).exp();
                }

                let log_sum_exp = max + sum.ln();

                log_sum_exp - row[label as usize]
            })
            .sum::<f32>(),
    };

    (log_loss + reg) * inv_sample_weights_sum
}

pub fn compute_gradient_multiclass_nointercept(
    x: ArrayView2<f32>,
    x_t: ArrayView2<f32>,
    y: ArrayView1<u32>,
    w: ArrayView1<f32>,
    n_classes: u32,
    n_features: u32,
    l1_reg: f32,
    l2_reg: f32,
    inv_sample_weights_sum: f32,
    sample_weights: Option<ArrayView1<f32>>,
) -> Array1<f32> {
    let mut scores = x.dot(
        &w.to_shape((n_features as usize, n_classes as usize))
            .unwrap(),
    );

    let mut loss_grad = match sample_weights {
        Some(weights) => {
            scores
                .axis_iter_mut(Axis(0))
                .into_par_iter()
                .zip(y.as_slice().unwrap())
                .zip(weights.as_slice().unwrap())
                .for_each(|((mut row, &label), &sw)| {
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
                    for v in row.iter_mut() {
                        *v /= sum;
                    }
                    row[label as usize] -= 1.0;
                    row *= sw;
                });
            x_t.dot(&scores)
        }

        None => {
            scores
                .axis_iter_mut(Axis(0))
                .into_par_iter()
                .zip(y.as_slice().unwrap())
                .for_each(|(mut row, &label)| {
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
                    for v in row.iter_mut() {
                        *v /= sum;
                    }
                    row[label as usize] -= 1.0;
                });
            x_t.dot(&scores)
        }
    };

    for (g, &wi) in loss_grad.iter_mut().zip(w.iter()) {
        *g += l1_reg * wi.signum() + l2_reg * wi;
        *g *= inv_sample_weights_sum;
    }
    loss_grad
        .into_shape_with_order(n_features as usize * n_classes as usize)
        .unwrap()
}

pub fn compute_gradient_multiclass_intercept(
    x: ArrayView2<f32>,
    x_t: ArrayView2<f32>,
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

    let (bias, w_base) = (w.slice(s![..n_classes]), w.slice(s![n_classes..]));

    let mut scores = x.dot(&w_base.to_shape((n_features, n_classes)).unwrap());

    scores += &bias;

    match sample_weights {
        Some(weights) => {
            scores
                .axis_iter_mut(Axis(0))
                .into_par_iter()
                .zip(y.as_slice().unwrap())
                .zip(weights.as_slice().unwrap())
                .for_each(|((mut row, &label), &sw)| {
                    let mut max = f32::NEG_INFINITY;

                    for &v in row.iter() {
                        if v > max {
                            max = v;
                        }
                    }

                    let mut sum = 0.0;

                    for v in row.iter_mut() {
                        *v = (*v - max).exp();
                        sum += *v;
                    }

                    let inv_sum = 1.0 / sum;

                    for v in row.iter_mut() {
                        *v *= inv_sum;
                    }

                    row[label as usize] -= 1.0;
                    row *= sw;
                });
        }

        None => {
            scores
                .axis_iter_mut(Axis(0))
                .into_par_iter()
                .zip(y.as_slice().unwrap())
                .for_each(|(mut row, &label)| {
                    let mut max = f32::NEG_INFINITY;

                    for &v in row.iter() {
                        if v > max {
                            max = v;
                        }
                    }

                    let mut sum = 0.0;

                    for v in row.iter_mut() {
                        *v = (*v - max).exp();
                        sum += *v;
                    }

                    let inv_sum = 1.0 / sum;

                    for v in row.iter_mut() {
                        *v *= inv_sum;
                    }

                    row[label as usize] -= 1.0;
                });
        }
    }

    let mut grad = Array1::<f32>::zeros(w.len());

    // bias gradient directly into first part
    let mut bias_grad = grad.slice_mut(s![..n_classes]);

    for (dst, src) in bias_grad.iter_mut().zip(scores.sum_axis(Axis(0))) {
        *dst = src * inv_sample_weights_sum;
    }

    // weight gradient directly into second part
    let mut weight_grad = grad.slice_mut(s![n_classes..]);

    let weight_grad_matrix = x_t.dot(&scores);

    for ((dst, &g), &wi) in weight_grad
        .iter_mut()
        .zip(weight_grad_matrix.iter())
        .zip(w_base.iter())
    {
        *dst = (g + l1_reg * wi.signum() + l2_reg * wi) * inv_sample_weights_sum;
    }

    grad
}
