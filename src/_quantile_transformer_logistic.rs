use ndarray::{Array1, ArrayView1};

pub fn nanpercentile(
    x: ArrayView1<f64>,
    references: ArrayView1<f64>,
) -> Array1<f64> {
    let mut quantiles = Array1::<f64>::zeros(references.len());

    // Collect non-NaN values
    let mut buffer = Vec::<f64>::with_capacity(x.len());

    for &v in x.iter() {
        if !v.is_nan() {
            buffer.push(v);
        }
    }

    if buffer.is_empty() {
        quantiles.fill(f64::NAN);
        return quantiles;
    }

    buffer.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    for (out, &q) in quantiles.iter_mut().zip(references.iter()) {
        *out = percentile_sorted(&buffer, q);
    }

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