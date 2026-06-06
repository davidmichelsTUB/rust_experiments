use rayon::prelude::*;

pub fn compute_minmax_scale_fit<const N: usize>(
    x: ndarray::ArrayView2<f32>,
) -> (ndarray::Array1<f32>, ndarray::Array1<f32>) {
    // Standard row-major Array2 is contiguous, so this is Some(..).
    let data = x
        .as_slice()
        .expect("array must be contiguous (C/row-major)");

    let (min, max) = data
        .par_chunks(N * 8192) // each task handles ~8192 rows; multiple of N => clean rows
        .map(|block| {
            let mut min = [f32::INFINITY; N];
            let mut max = [f32::NEG_INFINITY; N];
            for row in block.chunks_exact(N) {
                let row: &[f32; N] = row.try_into().unwrap(); // tells compiler the length
                for j in 0..N {
                    min[j] = min[j].min(row[j]);
                    max[j] = max[j].max(row[j]);
                }
            }
            (min, max)
        })
        .reduce(
            || ([f32::INFINITY; N], [f32::NEG_INFINITY; N]),
            |(mut amin, mut amax), (bmin, bmax)| {
                for j in 0..N {
                    amin[j] = amin[j].min(bmin[j]);
                    amax[j] = amax[j].max(bmax[j]);
                }
                (amin, amax)
            },
        );

    (
        ndarray::Array1::from(min.to_vec()),
        ndarray::Array1::from(max.to_vec()),
    )
}
