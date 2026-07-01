use ndarray::parallel::prelude::*;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use ndarray_stats::QuantileExt;

pub fn dot_argmax(x: ArrayView2<f32>, w: ArrayView1<f32>) -> Array1<u32> {
    let n_features = (x.shape()[1]) as u32;
    let n_classes = (w.len() / n_features as usize) as u32;

    Array1::from_vec(
        x.dot(
            &w.to_shape((n_features as usize, n_classes as usize))
                .unwrap(),
        )
        .axis_iter(Axis(0))
        .into_par_iter()
        .map(|row| row.argmax().unwrap() as u32)
        .collect::<Vec<u32>>(),
    )
}

pub fn dot_softmax(x: ArrayView2<f32>, w: ArrayView1<f32>) -> Array2<f32> {
    let n_features = x.shape()[1];
    let n_classes = w.len() / n_features;

    let mut score = x.dot(&w.to_shape((n_features, n_classes)).unwrap());
    score
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .for_each(|mut row| {
            let max = *row.max().unwrap();
            row.mapv_inplace(|x| (x - max).exp());
            let sum = row.sum();
            row.mapv_inplace(|x| x / sum);
        });
    score
}

pub fn compute_loss_multiclass(
    x: ArrayView2<f32>,
    y: ArrayView1<u32>,
    w: ArrayView1<f32>,
    n_classes: u32,
    n_features: u32,
    l1_reg: f32,
    l2_reg: f32,
    sample_weights_sum: f32,
    sample_weights: Option<ArrayView1<f32>>,
) -> f32 {
    let scores = x.dot(
        &w.to_shape((n_features as usize, n_classes as usize))
            .unwrap(),
    );

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

pub fn compute_gradient_multiclass(
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
    let mut scores = x.dot(
        &w.to_shape((n_features as usize, n_classes as usize))
            .unwrap(),
    );

    let ( loss_grad, mut reg_grad) = rayon::join(
        || {
            match sample_weights {
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
    reg_grad = (reg_grad + &loss_grad.to_shape(n_features as usize * n_classes as usize).unwrap()) / sample_weights_sum;
    reg_grad
}
