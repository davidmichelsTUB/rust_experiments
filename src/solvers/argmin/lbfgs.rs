use ndarray::{Array1, ArrayView1, ArrayView2};
use argmin::core::{CostFunction, Gradient, Executor, Error};
use argmin::solver::lbfgs::LBFGS;
use argmin::solver::linesearch::MoreThuenteLineSearch;
use crate::{compute_loss, compute_gradient};

struct LogisticProblem<'a> {
    x: ArrayView2<'a, f32>,
    y: ArrayView1<'a, f32>,
    l1_reg: f32,
    l2_reg: f32,
}

impl CostFunction for LogisticProblem<'_> {
    type Param = Array1<f32>;
    type Output = f32;
    fn cost(&self, w: &Array1<f32>) -> Result<f32, argmin::core::Error> {
        Ok(compute_loss(self.x, self.y, w.view(), self.l1_reg, self.l2_reg))
    }
}

impl Gradient for LogisticProblem<'_> {
    type Param = Array1<f32>;
    type Gradient = Array1<f32>;
    fn gradient(&self, w: &Array1<f32>) -> Result<Array1<f32>, argmin::core::Error> {
        Ok(compute_gradient(self.x, self.y, w.view(), self.l1_reg, self.l2_reg))
    }
}

////////////////////////////////////////////////////////////

pub fn fit(
    x: ArrayView2<f32>,
    y: ArrayView1<f32>,
    w_init: ArrayView1<f32>,
    l1_reg: f32,
    l2_reg: f32,
    max_iters: u64,
    m: usize
) -> Result<Array1<f32>, Error> {
    let problem = LogisticProblem { x, y, l1_reg, l2_reg };
    let linesearch = MoreThuenteLineSearch::new();

    let solver = LBFGS::new(linesearch, m);
    
    let res = Executor::new(problem, solver)
        .configure(|state| state.param(w_init.to_owned()).max_iters(max_iters))
        .run()?;

    Ok(res.state.best_param.ok_or(Error::msg("No solution found"))?)
}