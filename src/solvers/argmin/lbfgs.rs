use ndarray::{Array1, ArrayView1, ArrayView2};
use argmin::core::{CostFunction, Gradient, Executor, Error};
use argmin::solver::quasinewton::LBFGS;
use argmin::solver::linesearch::MoreThuenteLineSearch;
use crate::{compute_loss_binary, compute_new_gradient_binary, compute_loss_multiclass, compute_gradient_multiclass};

struct BinaryLogisticProblem<'a> {
    x: ArrayView2<'a, f32>,
    y: ArrayView1<'a, u32>,
    intercept: bool,
    sample_weights: Option<ArrayView1<'a, f32>>,
    l1_reg: f32,
    l2_reg: f32,
}


impl<'a> CostFunction for BinaryLogisticProblem<'a> {
    type Param = Array1<f32>;
    type Output = f32;

    fn cost(&self, w: &Array1<f32>) -> Result<f32, argmin::core::Error> {
        Ok(compute_loss_binary(self.x, self.y, w.view(), self.intercept, self.l1_reg, self.l2_reg, match self.sample_weights {
            Some(weights) => weights.sum() as f32,
            None => self.x.nrows() as f32,
        }, self.sample_weights))
    }
}

impl<'a> Gradient for BinaryLogisticProblem<'a> {
    type Param = Array1<f32>;
    type Gradient = Array1<f32>;
    fn gradient(&self, w: &Array1<f32>) -> Result<Array1<f32>, argmin::core::Error> {
        Ok(compute_new_gradient_binary(self.x, self.y, w.view(), self.intercept, self.l1_reg, self.l2_reg, match self.sample_weights {
            Some(weights) => weights.sum() as f32,
            None => self.x.nrows() as f32,
        }, self.sample_weights))
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
    sample_weights: Option<ArrayView1<'a, f32>>
) -> Result<Array1<f32>, Error> {

    let problem = BinaryLogisticProblem { x, y, l1_reg, l2_reg, intercept, sample_weights };
    let linesearch = MoreThuenteLineSearch::new();

    let w_init = Array1::<f32>::zeros(x.ncols() + if intercept { 1 } else { 0 });

    let solver = LBFGS::new(linesearch, m).with_tolerance_grad(tolerance)?;
    
    let res = Executor::new(problem, solver)
        .configure(|state| state.param(w_init).max_iters(max_iters))
        .run()?;

    Ok(res.state.best_param.ok_or(Error::msg("No solution found"))?)
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
}

impl<'a> CostFunction for MultiClassLogisticProblem<'a> {
    type Param = Array1<f32>;
    type Output = f32;
    fn cost(&self, w: &Array1<f32>) -> Result<f32, argmin::core::Error> {
        Ok(compute_loss_multiclass(self.x, self.y, w.view(), self.n_classes, self.n_features, self.l1_reg, self.l2_reg, match self.sample_weights {
            Some(weights) => weights.sum() as f32,
            None => self.x.nrows() as f32,
        }, self.sample_weights))
    }
}

impl<'a> Gradient for MultiClassLogisticProblem<'a> {
    type Param = Array1<f32>;
    type Gradient = Array1<f32>;
    fn gradient(&self, w: &Array1<f32>) -> Result<Array1<f32>, argmin::core::Error> {
        Ok(compute_gradient_multiclass(self.x, self.y, w.view(), self.n_classes, self.n_features, self.l1_reg, self.l2_reg, match self.sample_weights {
            Some(weights) => weights.sum() as f32,
            None => self.x.nrows() as f32,
        }, self.sample_weights))
    }
}

pub fn fit_multiclass<'a>(
    x: ArrayView2<'a, f32>,
    y: ArrayView1<'a, u32>,
    l1_reg: f32,
    l2_reg: f32,
    n_classes: u32,
    n_features: u32,
    max_iters: u64,
    m: usize,
    tolerance: f32,
    sample_weights: Option<ArrayView1<'a, f32>>,
) -> Result<Array1<f32>, Error> {

    let problem = MultiClassLogisticProblem { x, y, l1_reg, l2_reg, sample_weights, n_classes, n_features };
    let linesearch = MoreThuenteLineSearch::new();

    let w_init = Array1::<f32>::zeros(n_features as usize * n_classes as usize);

    let solver = LBFGS::new(linesearch, m).with_tolerance_grad(tolerance)?;
    
    let res = Executor::new(problem, solver)
        .configure(|state| state.param(w_init).max_iters(max_iters))
        .run()?;

    Ok(res.state.best_param.ok_or(Error::msg("No solution found"))?)
}