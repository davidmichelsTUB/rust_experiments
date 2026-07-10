use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

mod _binary_logistic;
mod _multi_class_logistic;
mod _quantile_transformer_logistic;
pub use _binary_logistic::{
    compute_loss_binary_intercept, compute_loss_binary_no_intercept, compute_new_gradient_binary_with_intercept, compute_new_gradient_binary_no_intercept, dot_argmax_binary, dot_sigmoid_fused_parallel,
};
pub use _multi_class_logistic::{
    compute_gradient_multiclass_intercept, compute_gradient_multiclass_nointercept,
    compute_loss_multiclass_intercept, compute_loss_multiclass_nointercept, dot_argmax_multiclass_intercept, dot_softmax_multiclass_intercept, dot_argmax_multiclass_nointercept, dot_softmax_multiclass_nointercept,
};
pub use _quantile_transformer_logistic::nanpercentile;

mod solvers;

// Logistic Regression Functions

#[pyfunction]
fn binary_predict<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f32>,
    w: PyReadonlyArray1<f32>,
    bias: f32,
    intercept: bool,
) -> Bound<'py, PyArray1<u32>> {
    dot_argmax_binary(x.as_array(), w.as_array(),  bias, intercept).into_pyarray_bound(py)
}

#[pyfunction]
fn binary_predict_proba<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f32>,
    w: PyReadonlyArray1<f32>,
    bias: f32,
    intercept: bool,
) -> Bound<'py, PyArray2<f32>> {
    dot_sigmoid_fused_parallel(x.as_array(), w.as_array(), bias, intercept).into_pyarray_bound(py)
}

#[pyfunction]
#[pyo3(signature = (x, y, l1_reg, l2_reg, intercept, max_iters, m, tolerance, sample_weights=None))]
fn binary_lbfgs_fit<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f32>,
    y: PyReadonlyArray1<u32>,
    l1_reg: f32,
    l2_reg: f32,
    intercept: bool,
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
        intercept,
        max_iters,
        m,
        tolerance,
        sample_weights_view,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(w.into_pyarray_bound(py))
}
///////////////////////////////////////////
#[pyfunction]
#[pyo3(signature = (x, w, intercept, bias=None))]
fn multiclass_predict_proba<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f32>,
    w: PyReadonlyArray2<f32>,
    intercept: bool,
    bias: Option<PyReadonlyArray1<f32>>,
) -> Bound<'py, PyArray2<f32>> {
    match intercept {
        true => {
            let bias_view = match bias {
                Some(ref b) => Some(b.as_array()),
                None => None,
            };
            dot_softmax_multiclass_intercept(x.as_array(), w.as_array(), bias_view.unwrap()).into_pyarray_bound(py)
        }
        false => dot_softmax_multiclass_nointercept(x.as_array(), w.as_array()).into_pyarray_bound(py)
    }
}

#[pyfunction]
#[pyo3(signature = (x, w, intercept, bias=None))]
fn multiclass_predict<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f32>,
    w: PyReadonlyArray2<f32>,
    intercept: bool,
    bias: Option<PyReadonlyArray1<f32>>
) -> Bound<'py, PyArray1<u32>> {

    match intercept {
        true => {
            let bias_view = match bias {
                Some(ref b) => Some(b.as_array()),
                None => None,
            };
            dot_argmax_multiclass_intercept(x.as_array(), w.as_array(), bias_view.unwrap()).into_pyarray_bound(py)
        }
        false => dot_argmax_multiclass_nointercept(x.as_array(), w.as_array()).into_pyarray_bound(py),
    }
}

#[pyfunction]
#[pyo3(signature = (x, y, l1_reg, l2_reg, n_classes, n_features, intercept, max_iters, m, tolerance, sample_weights=None))]
fn multiclass_lbfgs_fit<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f32>,
    y: PyReadonlyArray1<u32>,
    l1_reg: f32,
    l2_reg: f32,
    n_classes: u32,
    n_features: u32,
    intercept: bool,
    max_iters: u64,
    m: usize,
    tolerance: f32,
    sample_weights: Option<PyReadonlyArray1<f32>>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let sample_weights_view = match sample_weights {
        Some(ref w) => Some(w.as_array()),
        None => None,
    };
    let w = solvers::argmin::lbfgs::fit_multiclass(
        x.as_array(),
        y.as_array(),
        l1_reg,
        l2_reg,
        n_classes,
        n_features,
        intercept,
        max_iters,
        m,
        tolerance,
        sample_weights_view,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(w.into_pyarray_bound(py))
}

// Quantile Transformer

#[pyfunction]
fn rust_fit_dense<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    references: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {

    let x_asarr = x.as_array();
    let references_asarr = references.as_array();

    let quantiles = _quantile_transformer_logistic::nanpercentile(
        x_asarr,
        references_asarr,
    );
    Ok(quantiles.into_pyarray_bound(py))
}

// Here where the python modulation comes in
#[pymodule]
fn rust_logistic(m: &Bound<'_, PyModule>) -> PyResult<()> {

    // Logistic Regression Functions
    m.add_function(wrap_pyfunction!(binary_predict, m)?)?;
    m.add_function(wrap_pyfunction!(binary_predict_proba, m)?)?;
    m.add_function(wrap_pyfunction!(binary_lbfgs_fit, m)?)?;
    m.add_function(wrap_pyfunction!(multiclass_lbfgs_fit, m)?)?;
    m.add_function(wrap_pyfunction!(multiclass_predict_proba, m)?)?;
    m.add_function(wrap_pyfunction!(multiclass_predict, m)?)?;

    // Quantile Transformer Functions
    m.add_function(wrap_pyfunction!(rust_fit_dense, m)?)?;
    Ok(())
}

//////////////////////////////////////////
