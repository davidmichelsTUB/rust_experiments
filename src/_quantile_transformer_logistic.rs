// use ndarray::parallel::prelude::*;
// use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut2, Axis};
// use ndarray_stats::QuantileExt;
// use ndarray_interp::interp1d;
// use stack_stack::Stack;


// pub fn dense_transform_forward(
//     mut x_no_na: ArrayViewMut2<f32>,
//     quantile_values: ArrayView2<f32>,
//     output_distribution: &str,
    
// ) -> ArrayViewMut2<f32> {
//     let last_q = quantile_values.shape()[0] - 1;
//     let lower_bound_y: f32 = 0.0;
//     let upper_bound_y: f32 = 1.0;

//     x_no_na.axis_iter_mut(Axis(1))
//         .into_par_iter()
//         .zip(quantile_values.axis_iter(Axis(1)).into_par_iter())
//         .for_each(|(mut col, quantile)| {
//             let lower_bound_x = quantile[0];
//             let upper_bound_x = quantile[last_q];



//             for (i, val) in col.indexed_iter_mut() {

//             }

//             col.mapv_inplace(|val| {
                
//                 match val == lower_bound_x {
//                     true => 
//                     _ =>
//                 }
                
//                 // get the places, where x matches the bounds
//                 val
//             });
//         });
//         x_no_na
// }
