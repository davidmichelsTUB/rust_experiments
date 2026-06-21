use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

mod _binary_logstic;
pub use _binary_logstic::{compute_gradient, compute_loss, predict_proba_impl};

mod solvers;
use solvers::argmin::lbfgs;

#[pyfunction]
fn binary_predict_proba<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f32>,
    w: PyReadonlyArray1<f32>,
    kernel: &str,
) -> Bound<'py, PyArray1<f32>> {
    predict_proba_impl(x.as_array(), w.as_array(), kernel).into_pyarray_bound(py)
}

#[pyfunction]
#[pyo3(signature = (x, y, l1_reg, l2_reg, max_iters, m, sample_weights=None))]
fn binary_lbfgs_fit<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f32>,
    y: PyReadonlyArray1<f32>,
    l1_reg: f32,
    l2_reg: f32,
    max_iters: u64,
    m: usize,
    sample_weights: Option<PyReadonlyArray1<f32>>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let sample_weights_view = match sample_weights {
        Some(ref w) => Some(w.as_array()),
        None => None,
    };    
    let w = lbfgs::fit(
        x.as_array(),
        y.as_array(),
        l1_reg,
        l2_reg,
        max_iters,
        m,
        sample_weights_view,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(w.into_pyarray_bound(py))
}

// Here where the python modulation comes in
#[pymodule]
fn rust_logistic(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(binary_predict_proba, m)?)?;
    m.add_function(wrap_pyfunction!(binary_lbfgs_fit, m)?)?;
    Ok(())
}
