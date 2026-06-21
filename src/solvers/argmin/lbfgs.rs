use ndarray::{Array1, ArrayView1, ArrayView2};
use argmin::core::{CostFunction, Gradient, Executor, Error};
use argmin::solver::quasinewton::LBFGS;
use argmin::solver::linesearch::MoreThuenteLineSearch;
use crate::{compute_loss, compute_gradient};

struct LogisticProblem<'a> {
    x: ArrayView2<'a, f32>,
    y: ArrayView1<'a, f32>,
    sample_weights: Option<ArrayView1<'a, f32>>,
    l1_reg: f32,
    l2_reg: f32,
}

impl<'a> CostFunction for LogisticProblem<'a> {
    type Param = Array1<f32>;
    type Output = f32;
    fn cost(&self, w: &Array1<f32>) -> Result<f32, argmin::core::Error> {
        Ok(compute_loss(self.x, self.y, w.view(), self.l1_reg, self.l2_reg, self.sample_weights))
    }
}

impl<'a> Gradient for LogisticProblem<'a> {
    type Param = Array1<f32>;
    type Gradient = Array1<f32>;
    fn gradient(&self, w: &Array1<f32>) -> Result<Array1<f32>, argmin::core::Error> {
        Ok(compute_gradient(self.x, self.y, w.view(), self.l1_reg, self.l2_reg, self.sample_weights))
    }
}

////////////////////////////////////////////////////////////

pub fn fit<'a>(
    x: ArrayView2<'a, f32>,
    y: ArrayView1<'a, f32>,
    l1_reg: f32,
    l2_reg: f32,
    max_iters: u64,
    m: usize,
    sample_weights: Option<ArrayView1<'a, f32>>
) -> Result<Array1<f32>, Error> {
    let problem = LogisticProblem { x, y, l1_reg, l2_reg, sample_weights };
    let linesearch = MoreThuenteLineSearch::new();

    let w_init = Array1::<f32>::zeros(x.ncols());

    let solver = LBFGS::new(linesearch, m);
    
    let res = Executor::new(problem, solver)
        .configure(|state| state.param(w_init).max_iters(max_iters))
        .run()?;

    Ok(res.state.best_param.ok_or(Error::msg("No solution found"))?)
}