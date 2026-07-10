use ndarray::parallel::prelude::*;
use ndarray::{Array2, ArrayView1, ArrayView2, Axis};


pub fn nanpercentile(
    x: ArrayView2<f64>,
    references: ArrayView1<f64>,
) -> Array2<f64> {
    let n_features = x.ncols();

    let mut quantiles = Array2::<f64>::zeros((references.len(), n_features));

    let x_rows = x.nrows();

    quantiles
        .axis_iter_mut(Axis(1))
        .into_par_iter()
        .enumerate()
        .for_each(|(feature, mut output)| {
            let mut buffer = Vec::<f64>::with_capacity(x_rows);

            // extract column
            for row in 0..x_rows {
                let v = x[(row, feature)];
                if !v.is_nan() {
                    buffer.push(v);
                }
            }

            if buffer.is_empty() {
                output.fill(f64::NAN);
                return;
            }

            buffer.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

            for (i, &q) in references.iter().enumerate() {
                output[i] = percentile_sorted(&buffer, q);
            }
        });

    quantiles
}


#[inline(always)]
fn percentile_sorted(values: &[f64], q: f64) -> f64 {
    let n = values.len();

    if n == 1 {
        return values[0];
    }

    let pos = q * (n as f64 - 1.0);

    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;

    if lo == hi {
        values[lo]
    } else {
        let weight = pos - lo as f64;
        values[lo] * (1.0 - weight) + values[hi] * weight
    }
}