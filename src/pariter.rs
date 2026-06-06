use ndarray::Axis;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

pub fn compute_minmax_scale_fit(
    x: ndarray::ArrayView2<f32>,
    n_chunks: usize,
) -> (ndarray::Array1<f32>, ndarray::Array1<f32>) {
    let (n_rows, n_cols) = x.dim();
    let chunk_size = (n_rows / n_chunks).max(1);

    let compute = || {
        x.axis_chunks_iter(Axis(0), chunk_size)
            .into_par_iter()
            .map(|chunk| {
                let mut min = vec![f32::INFINITY; n_cols];
                let mut max = vec![f32::NEG_INFINITY; n_cols];
                for row in chunk.rows() {
                    for col_idx in 0..n_cols {
                        let value = row[col_idx];
                        if min[col_idx] > value {
                            min[col_idx] = value
                        }
                        if max[col_idx] < value {
                            max[col_idx] = value
                        }
                    }
                }
                (min, max)
            })
            .reduce(
                || (vec![f32::INFINITY; n_cols], vec![f32::NEG_INFINITY; n_cols]),
                |(mut amin, mut amax), (bmin, bmax)| {
                    for col in 0..n_cols {
                        amin[col] = amin[col].min(bmin[col]);
                        amax[col] = amax[col].max(bmax[col]);
                    }

                    (amin, amax)
                },
            )
    };

    let (min, max) = compute();
    return (ndarray::Array1::from(min), ndarray::Array1::from(max));
}

pub fn compute_minmax_scale_transform(
    x: ndarray::ArrayView2<f32>,
    min: ndarray::Array1<f32>,
    max: ndarray::Array1<f32>,
    n_chunks: usize,
) -> ndarray::Array2<f32> {
    let (n_rows, n_cols) = x.dim();
    let chunk_size = (n_rows / n_chunks).max(1);
    let mut out = ndarray::Array2::<f32>::zeros((n_rows, n_cols));

    out.axis_chunks_iter_mut(Axis(0), chunk_size)
        .into_par_iter()
        .zip(x.axis_chunks_iter(Axis(0), chunk_size).into_par_iter())
        .for_each(|(mut out_chunk, in_chunk)| {
            for (mut out_row, in_row) in out_chunk.rows_mut().into_iter().zip(in_chunk.rows()) {
                for col_idx in 0..n_cols {
                    out_row[col_idx] =
                        (in_row[col_idx] - min[col_idx]) / (max[col_idx] - min[col_idx])
                }
            }
        });

    out
}

#[pyfunction]
#[pyo3(signature = (x, n_chunks))]
pub fn minmax_scale_fit(
    py: Python<'_>,
    x: PyReadonlyArray2<f32>,
    n_chunks: usize,
) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<f32>>)> {
    if n_chunks == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "n_chunks must be >= 1",
        ));
    }
    let x_view = x.as_array();
    let (min, max) = py.detach(|| compute_minmax_scale_fit(x_view, n_chunks));
    let py_mean = min.into_pyarray(py).to_owned();
    let py_scale = max.into_pyarray(py).to_owned();
    Ok((Py::from(py_mean), Py::from(py_scale)))
}
