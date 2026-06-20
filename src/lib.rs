use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};

mod _binary_logstic;
pub use _binary_logstic::{compute_loss, compute_gradient, predict_proba_impl};

mod solvers;
use solvers::argmin::lbfgs;

#[pyfunction]
fn predict_proba<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f32>,
    w: PyReadonlyArray1<f32>,
    kernel: &str,
) -> Bound<'py, PyArray1<f32>> {
    predict_proba_impl(x.as_array(), w.as_array(), kernel).into_pyarray_bound(py)
}

#[pyfunction]
fn lbfgs_fit<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f32>,
    y: PyReadonlyArray1<f32>,
    l1_reg: f32,
    l2_reg: f32,
    max_iters: u64,
    m: usize,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let w = lbfgs::fit(x.as_array(), y.as_array(), l1_reg, l2_reg, max_iters, m)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(w.into_pyarray_bound(py))
}

#[pymodule]
fn rust_logistic(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(predict_proba, m)?)?;
    m.add_function(wrap_pyfunction!(lbfgs_fit, m)?)?;
    Ok(())
}
