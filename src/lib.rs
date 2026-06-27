use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

mod _binary_logistic;
mod _multi_class_logistic;
pub use _binary_logistic::{compute_new_gradient_binary, compute_loss_binary, predict_proba_impl_binary};
pub use _multi_class_logistic::{compute_loss_multiclass, compute_loss_and_gradient_multiclass, predict_impl_multiclass, predict_proba_impl_multiclass, scaled_add2d, softmax_inplace, dot2d};

mod solvers;

#[pyfunction]
fn binary_predict_proba<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f32>,
    w: PyReadonlyArray1<f32>,
    kernel: &str,
) -> Bound<'py, PyArray1<f32>> {
    predict_proba_impl_binary(x.as_array(), w.as_array(), kernel).into_pyarray_bound(py)
}

#[pyfunction]
fn multiclass_predict_proba<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f32>,
    w: PyReadonlyArray2<f32>,
) -> Bound<'py, PyArray2<f32>> {
    predict_proba_impl_multiclass(x.as_array(), w.as_array()).into_pyarray_bound(py)
}

#[pyfunction]
fn multiclass_predict<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f32>,
    w: PyReadonlyArray2<f32>,
) -> Bound<'py, PyArray1<usize>> {
    predict_impl_multiclass(x.as_array(), w.as_array()).into_pyarray_bound(py)
}

#[pyfunction]
#[pyo3(signature = (x, y, l1_reg, l2_reg, max_iters, m, tolerance, sample_weights=None))]
fn binary_lbfgs_fit<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f32>,
    y: PyReadonlyArray1<f32>,
    l1_reg: f32,
    l2_reg: f32,
    max_iters: u64,
    m: usize,
    tolerance: f32,
    sample_weights: Option<PyReadonlyArray1<f32>>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let sample_weights_view = match sample_weights {
        Some(ref w) => Some(w.as_array()),
        None => None,
    };    
    let w = solvers::argmin::lbfgs::fit_binary(
        x.as_array(),
        y.as_array(),
        l1_reg,
        l2_reg,
        max_iters,
        m,
        tolerance,
        sample_weights_view,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(w.into_pyarray_bound(py))
}

#[pyfunction]
#[pyo3(signature = (x, y, n_classes, l1_reg, l2_reg, max_iters, m, tolerance, sample_weights=None))]
fn multiclass_lbfgs_fit<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f32>,
    y: PyReadonlyArray1<usize>,
    n_classes: usize,
    l1_reg: f32,
    l2_reg: f32,
    max_iters: u64,
    m: usize,
    tolerance: f32,
    sample_weights: Option<PyReadonlyArray1<f32>>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let sample_weights_view = match sample_weights {
        Some(ref w) => Some(w.as_array()),
        None => None,
    };
    let w = solvers::own_multiclass::lbfgs::fit_multiclass(
        x.as_array(),
        y.as_array(),
        n_classes,
        l1_reg,
        l2_reg,
        max_iters,
        m,
        tolerance,
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
    m.add_function(wrap_pyfunction!(multiclass_predict_proba, m)?)?;
    m.add_function(wrap_pyfunction!(multiclass_predict, m)?)?;
    m.add_function(wrap_pyfunction!(multiclass_lbfgs_fit, m)?)?;
    Ok(())
}
