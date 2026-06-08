use ndarray::Axis;
pub fn compute_minmax_scale_fit(
    x: ndarray::ArrayView2<f32>,
    n_chunks: usize,
) -> (ndarray::Array1<f32>, ndarray::Array1<f32>) {
    let (n_rows, n_cols) = x.dim();
    let chunk_size = (n_rows / n_chunks).max(1);
    
    let compute = || {
        let partials: Vec<(Vec<f32>, Vec<f32>)> = x
            .axis_chunks_iter(Axis(0), chunk_size)
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
            .collect();

        let mut total_min = vec![f32::INFINITY; n_cols];
        let mut total_max = vec![f32::NEG_INFINITY; n_cols];

        for col_idx in 0..n_cols {
            for (mins, maxs) in partials.iter() {
                if total_min[col_idx] > mins[col_idx] {
                    total_min[col_idx] = mins[col_idx]
                }
                if total_max[col_idx] < maxs[col_idx] {
                    total_max[col_idx] = maxs[col_idx]
                }
            }
        }
        (total_min, total_max)
    };

    let (min, max) = compute();
    return (ndarray::Array1::from(min), ndarray::Array1::from(max));
}
