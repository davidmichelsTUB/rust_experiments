use pyo3::prelude::*;
mod pariter;

#[pymodule]
fn rust_backend(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pariter::minmax_scale_fit, m)?)?;
    m.add_function(wrap_pyfunction!(pariter::minmax_scale_transform, m)?)?;
    Ok(())
}
