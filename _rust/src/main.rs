use ndarray::Array2;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

use std::time::Instant;
mod pariter;

fn main() {
    let x = Array2::<f32>::random((10_000_000, 5), Uniform::new(4., 10.).unwrap());
    let x_view = x.view();
    const N_CHUNKS: usize = 255;
    
    let start = Instant::now();
    let (min_vals, max_vals) = pariter::compute_minmax_scale_fit(x_view, N_CHUNKS);
    println!("{:?}", (&min_vals, &max_vals));

    let x_transformed =
        pariter::compute_minmax_scale_transform(x_view, min_vals.view(), max_vals.view(), N_CHUNKS);
    let (min_vals_test, max_vals_test) =
        pariter::compute_minmax_scale_fit(x_transformed.view(), N_CHUNKS);

    println!("{:?}", (min_vals_test, max_vals_test));
    let duration = start.elapsed();
    println!("Time{duration:?}")
}
