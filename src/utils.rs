use ndarray::prelude::*;
use std::time::Instant;

pub fn get_ms(time: Instant) -> f64 {
    (time.elapsed().as_nanos() as f64) / 1_000_000_f64
}

pub fn denoise(array: &Array1<f64>) -> Array1<f64> {
    let mut denoised_array = array.to_owned();
    let len = array.len();
    let gamma = 0.25;

    denoised_array[0] = array[0] + gamma * (array[1] - array[0]);
    for i in 1..(array.len() - 1) {
        denoised_array[i] = array[i] + gamma * (array[i - 1] + array[i + 1] - 2.0 * array[i]);
    }
    denoised_array[len - 1] = array[len - 1] + gamma * (array[len - 2] - array[len - 1]);

    denoised_array
}
