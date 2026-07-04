use ndarray::parallel::prelude::*;
use ndarray::s;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use ndarray_stats::QuantileExt;
use std::mem::MaybeUninit;
use std::ops::AddAssign;

pub fn dot_argmax_multiclass(
    x: ArrayView2<f32>,
    w: ArrayView1<f32>,
    intercept: bool,
    n_classes: u32,
) -> Array1<u32> {
    let n_features = x.shape()[1];
    let mut out = Array1::<u32>::uninit(x.shape()[0]);

    if !intercept {
        let w_reshaped = w.to_shape((n_features, n_classes as usize)).unwrap();

        out.as_slice_mut()
            .unwrap()
            .par_iter_mut()
            .zip(x.axis_iter(Axis(0)).into_par_iter())
            .for_each(|(out_i, x_row)| {
                let z = x_row.dot(&w_reshaped);
                let mut best_idx: u32 = 0;
                let mut best_val = z[0];

                for (i, &v) in z.iter().enumerate().skip(1) {
                    if v > best_val {
                        best_val = v;
                        best_idx = i as u32;
                    }
                }

                *out_i = MaybeUninit::new(best_idx);
            });
    } else {
        let (bias, w_base) = (
            w.slice(s![..n_classes as usize]),
            w.slice(s![n_classes as usize..]),
        );
        let w_base_reshaped = w_base.to_shape((n_features, n_classes as usize)).unwrap();

        out.as_slice_mut()
            .unwrap()
            .par_iter_mut()
            .zip(x.axis_iter(Axis(0)).into_par_iter())
            .for_each(|(out_i, x_row)| {
                let z = x_row.dot(&w_base_reshaped) + &bias;
                let mut best_idx: u32 = 0;
                let mut best_val = z[0];

                for (i, &v) in z.iter().enumerate().skip(1) {
                    if v > best_val {
                        best_val = v;
                        best_idx = i as u32;
                    }
                }
                *out_i = MaybeUninit::new(best_idx);
            });
    }

    unsafe { out.assume_init() }
}

pub fn dot_softmax_multiclass(
    x: ArrayView2<f32>,
    w: ArrayView1<f32>,
    intercept: bool,
    n_classes: u32,
) -> Array2<f32> {
    let n_features = x.shape()[1];

    let mut score = match intercept {
        true => {
            let (bias, w_base) = (
                w.slice(s![..n_classes as usize]),
                w.slice(s![n_classes as usize..]),
            );
            let w_base_reshaped = w_base.to_shape((n_features, n_classes as usize)).unwrap();
            x.dot(&w_base_reshaped) + &bias
        }

        false => {
            let w_reshaped = w.to_shape((n_features, n_classes as usize)).unwrap();
            x.dot(&w_reshaped)
        }
    };

    score
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .for_each(|mut row| {
            let max = *row.max().unwrap();

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

pub fn compute_loss_multiclass(
    x: ArrayView2<f32>,
    y: ArrayView1<u32>,
    w: ArrayView1<f32>,
    n_classes: u32,
    n_features: u32,
    intercept: bool,
    l1_reg: f32,
    l2_reg: f32,
    sample_weights_sum: f32,
    sample_weights: Option<ArrayView1<f32>>,
) -> f32 {

    let scores = match intercept {
        true => {
            let (bias, w_base) = (
                w.slice(s![..n_classes as usize]),
                w.slice(s![n_classes as usize..])
            );
            x.dot(&w_base.to_shape((n_features as usize, n_classes as usize)).unwrap()) + &bias
        }
        false => {
            let w_reshaped = w.to_shape((n_features as usize, n_classes as usize)).unwrap();
            x.dot(&w_reshaped)
        }
    };



    let (log_loss, reg_loss) = rayon::join(
        || match sample_weights {
            Some(weights) => scores
                .axis_iter(Axis(0))
                .into_par_iter()
                .zip(y.as_slice().unwrap())
                .zip(weights.as_slice().unwrap())
                .map(|((row, &label), &sw)| {
                    let max = *row.max().unwrap();

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
                    let max = row.max().unwrap();

                    let mut sum: f32 = 0.0;
                    for &v in row {
                        sum += (v - max).exp();
                    }

                    let log_sum_exp = max + sum.ln();

                    log_sum_exp - row[label as usize]
                })
                .sum::<f32>(),
        },
        || {
            w.as_slice()
                .unwrap()
                .par_iter()
                .map(|wi| l1_reg * wi.abs() + 0.5 * l2_reg * wi * wi)
                .sum::<f32>()
        },
    );
    (log_loss + reg_loss) / sample_weights_sum
}

pub fn compute_gradient_multiclass_nointercept(
    x: ArrayView2<f32>,
    y: ArrayView1<u32>,
    w: ArrayView1<f32>,
    n_classes: u32,
    n_features: u32,
    l1_reg: f32,
    l2_reg: f32,
    sample_weights_sum: f32,
    sample_weights: Option<ArrayView1<f32>>,
) -> Array1<f32> {
    let mut scores = x.dot(&w.to_shape((n_features as usize, n_classes as usize)).unwrap());


    let (loss_grad, mut reg_grad) = rayon::join(
        || match sample_weights {
            Some(weights) => {
                scores
                    .axis_iter_mut(Axis(0))
                    .into_par_iter()
                    .zip(y.as_slice().unwrap())
                    .zip(weights.as_slice().unwrap())
                    .for_each(|((mut row, &label), &sw)| {
                        let max = *row.max().unwrap();
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
                x.t().dot(&scores)
            }

            None => {
                scores
                    .axis_iter_mut(Axis(0))
                    .into_par_iter()
                    .zip(y.as_slice().unwrap())
                    .for_each(|(mut row, &label)| {
                        let max = *row.max().unwrap();
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
                x.t().dot(&scores)
            }
        },
        || {
            Array1::from_vec(
                w.as_slice()
                    .unwrap()
                    .par_iter()
                    .map(|wi| l1_reg * wi.signum() + l2_reg * wi)
                    .collect::<Vec<f32>>(),
            )
        },
    );
    
    reg_grad = (reg_grad
        + &loss_grad
            .to_shape(n_features as usize * n_classes as usize)
            .unwrap())
        / sample_weights_sum;
    reg_grad
}

pub fn compute_gradient_multiclass_intercept(
    x: ArrayView2<f32>,
    y: ArrayView1<u32>,
    w: ArrayView1<f32>,
    n_classes: u32,
    n_features: u32,
    l1_reg: f32,
    l2_reg: f32,
    sample_weights_sum: f32,
    sample_weights: Option<ArrayView1<f32>>,
) -> Array1<f32> {
    let (bias, w_base) = (
        w.slice(s![..n_classes as usize]),
        w.slice(s![n_classes as usize..]),
    );
    let mut scores = x.dot(&w_base.to_shape((n_features as usize, n_classes as usize)).unwrap())
        + &bias;

    let ((loss_weight_grad, loss_bias_grad), mut reg_grad) = rayon::join(
        || match sample_weights {
            Some(weights) => {
                scores
                    .axis_iter_mut(Axis(0))
                    .into_par_iter()
                    .zip(y.as_slice().unwrap())
                    .zip(weights.as_slice().unwrap())
                    .for_each(|((mut row, &label), &sw)| {
                        let max = *row.max().unwrap();
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
                (x.t().dot(&scores), scores.sum_axis(Axis(0)))
            }

            None => {
                scores
                    .axis_iter_mut(Axis(0))
                    .into_par_iter()
                    .zip(y.as_slice().unwrap())
                    .for_each(|(mut row, &label)| {
                        let max = *row.max().unwrap();
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
                (x.t().dot(&scores), scores.sum_axis(Axis(0)))
            }
        },
        || {
            Array1::from_vec(
                w.as_slice()
                    .unwrap()
                    .par_iter()
                    .map(|wi| l1_reg * wi.signum() + l2_reg * wi)
                    .collect::<Vec<f32>>(),
            )
        }
    );
    
    reg_grad.slice_mut(s![..n_classes as usize]).add_assign(&loss_bias_grad);
    reg_grad.slice_mut(s![n_classes as usize..]).add_assign(
        &loss_weight_grad
            .to_shape(n_features as usize * n_classes as usize)
            .unwrap(),
    );

    reg_grad / sample_weights_sum
}
