use ndarray::Axis;

use rayon::prelude::*;
pub fn compute_minmax_scale_fit<const N_CHUNKS: usize>(
    x: ndarray::ArrayView2<f32>,
) -> (ndarray::Array1<f32>, ndarray::Array1<f32>) {
    let (n_rows, n_cols) = x.dim();
    let chunk_size = (n_rows / N_CHUNKS).max(1);

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
