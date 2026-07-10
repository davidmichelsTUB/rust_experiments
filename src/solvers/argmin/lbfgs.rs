use crate::{
    compute_gradient_multiclass_intercept, compute_gradient_multiclass_nointercept, compute_loss_multiclass_intercept, compute_loss_multiclass_nointercept
};

use crate::{compute_loss_binary_intercept, compute_loss_binary_no_intercept, compute_new_gradient_binary_with_intercept, compute_new_gradient_binary_no_intercept};
use argmin::core::{CostFunction, Error, Executor, Gradient};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;
use ndarray::{Array1, ArrayView1, ArrayView2};

struct BinaryLogisticProblem<'a> {
    x: ArrayView2<'a, f32>,
    y: ArrayView1<'a, u32>,
    sample_weights: Option<ArrayView1<'a, f32>>,
    l1_reg: f32,
    l2_reg: f32,
    cost_fn: fn(
        ArrayView2<f32>,
        ArrayView1<u32>,
        ArrayView1<f32>,
        f32,
        f32,
        f32,
        Option<ArrayView1<f32>>,
    ) -> f32,
    grad_fn: fn(
        ArrayView2<f32>,
        ArrayView1<u32>,
        ArrayView1<f32>,
        f32,
        f32,
        f32,
        Option<ArrayView1<f32>>,
    ) -> Array1<f32>,
}

impl<'a> CostFunction for BinaryLogisticProblem<'a> {
    type Param = Array1<f32>;
    type Output = f32;

    fn cost(&self, w: &Array1<f32>) -> Result<f32, argmin::core::Error> {
        let sample_weights_sum = match self.sample_weights {
            Some(weights) => weights.sum(),
            None => self.x.nrows() as f32,
        };

        Ok((self.cost_fn)(
            self.x,
            self.y,
            w.view(),
            self.l1_reg,
            self.l2_reg,
            sample_weights_sum,
            self.sample_weights,
        ))
    }
}

impl<'a> Gradient for BinaryLogisticProblem<'a> {
    type Param = Array1<f32>;
    type Gradient = Array1<f32>;
    fn gradient(&self, w: &Array1<f32>) -> Result<Array1<f32>, argmin::core::Error> {

        let sample_weights_sum = match self.sample_weights {
            Some(weights) => weights.sum(),
            None => self.x.nrows() as f32,
        };

        Ok((self.grad_fn)(
            self.x,
            self.y,
            w.view(),
            self.l1_reg,
            self.l2_reg,
            sample_weights_sum,
            self.sample_weights,
        ))
    }
}
////////////////////////////////////////////////////////////

pub fn fit_binary<'a>(
    x: ArrayView2<'a, f32>,
    y: ArrayView1<'a, u32>,
    l1_reg: f32,
    l2_reg: f32,
    intercept: bool,
    max_iters: u64,
    m: usize,
    tolerance: f32,
    sample_weights: Option<ArrayView1<'a, f32>>,
) -> Result<Array1<f32>, Error> {
    let problem = BinaryLogisticProblem {
        x,
        y,
        l1_reg,
        l2_reg,
        sample_weights,
        cost_fn : match intercept {
            true => compute_loss_binary_intercept,
            false => compute_loss_binary_no_intercept,
        },
        grad_fn: match intercept {
            true => compute_new_gradient_binary_with_intercept,
            false => compute_new_gradient_binary_no_intercept,
        },
    };
    let linesearch = MoreThuenteLineSearch::new();

    let w_shape: usize = match intercept {
        true => x.ncols() + 1,
        false => x.ncols(),
    };

    let w_init = Array1::<f32>::zeros(w_shape);

    let solver = LBFGS::new(linesearch, m).with_tolerance_grad(tolerance)?;

    let res = Executor::new(problem, solver)
        .configure(|state| state.param(w_init).max_iters(max_iters))
        .run()?;

    Ok(res
        .state
        .best_param
        .ok_or(Error::msg("No solution found"))?)
}

////////////////////////////////////////////////////////////////
struct MultiClassLogisticProblem<'a> {
    x: ArrayView2<'a, f32>,
    y: ArrayView1<'a, u32>,
    sample_weights: Option<ArrayView1<'a, f32>>,
    l1_reg: f32,
    l2_reg: f32,
    n_classes: u32,
    n_features: u32,
    cost_fn: fn(
        ArrayView2<f32>,
        ArrayView1<u32>,
        ArrayView1<f32>,
        u32,
        u32,
        f32,
        f32,
        f32,
        Option<ArrayView1<f32>>,
    ) -> f32,
    grad_fn: fn(
        ArrayView2<f32>,
        ArrayView1<u32>,
        ArrayView1<f32>,
        u32,
        u32,
        f32,
        f32,
        f32,
        Option<ArrayView1<f32>>,
    ) -> Array1<f32>    
}

impl<'a> CostFunction for MultiClassLogisticProblem<'a> {
    type Param = Array1<f32>;
    type Output = f32;
    fn cost(&self, w: &Array1<f32>) -> Result<f32, argmin::core::Error> {
        let sample_weights_sum = match self.sample_weights {
            Some(weights) => weights.sum(),
            None => self.x.nrows() as f32,
        };

        Ok((self.cost_fn)(
            self.x,
            self.y,
            w.view(),
            self.n_classes,
            self.n_features,
            self.l1_reg,
            self.l2_reg,
            1.0 / sample_weights_sum,
            self.sample_weights,
        ))
    }
}

impl<'a> Gradient for MultiClassLogisticProblem<'a> {
    type Param = Array1<f32>;
    type Gradient = Array1<f32>;
    fn gradient(&self, w: &Array1<f32>) -> Result<Array1<f32>, argmin::core::Error> {
        
        let sample_weights_sum = match self.sample_weights {
            Some(weights) => weights.sum(),
            None => self.x.nrows() as f32,
        };

        Ok((self.grad_fn)(
            self.x,
            self.y,
            w.view(),
            self.n_classes,
            self.n_features,
            self.l1_reg,
            self.l2_reg,
            1.0 / sample_weights_sum,
            self.sample_weights,
        ))
    }
}

pub fn fit_multiclass<'a>(
    x: ArrayView2<'a, f32>,
    y: ArrayView1<'a, u32>,
    l1_reg: f32,
    l2_reg: f32,
    n_classes: u32,
    n_features: u32,
    intercept: bool,
    max_iters: u64,
    m: usize,
    tolerance: f32,
    sample_weights: Option<ArrayView1<'a, f32>>,
) -> Result<Array1<f32>, Error> {
    let problem = MultiClassLogisticProblem {
        x,
        y,
        l1_reg,
        l2_reg,
        sample_weights,
        n_classes,
        n_features,
        cost_fn: match intercept {
            true => compute_loss_multiclass_intercept,
            false => compute_loss_multiclass_nointercept,
        },
        grad_fn: match intercept {
            true => compute_gradient_multiclass_intercept,
            false => compute_gradient_multiclass_nointercept,
        },
    };
    let linesearch = MoreThuenteLineSearch::new();

    let w_shape: usize = match intercept {
        true => n_classes as usize + (n_features as usize * n_classes as usize),
        false => (n_features as usize) * (n_classes as usize),
    };
    let w_init = Array1::<f32>::zeros(w_shape);

    let solver = LBFGS::new(linesearch, m).with_tolerance_grad(tolerance)?;

    let res = Executor::new(problem, solver)
        .configure(|state| state.param(w_init).max_iters(max_iters))
        .run()?;

    Ok(res
        .state
        .best_param
        .ok_or(Error::msg("No solution found"))?)
}
