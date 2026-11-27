use ndarray::Array3;
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray3};
use pyo3::prelude::*;

fn rescale_intensity(x: f32, in_range_low: f32, in_range_high: f32, range: f32) -> u8 {
    //asssumes out_min = 0 and out_max = 255
    if x <= in_range_low {
        0
    } else if x >= in_range_high {
        255
    } else {
        (255.0 * ((x - in_range_low) / range)) as u8
    }
}

fn rust_contrast_enhancer(img: Array3<u8>, low_p: u8, high_p: u8) -> Array3<u8> {
    /*Enhance contrast of the input image using intensity adjustment.

    Values at or below the low percentile are saturated to 0, values at or
    above the high percentile are saturated to 255, and values in between are
    linearly rescaled. If the percentiles coincide, the image min and max are
    used instead.

    Args:
        img: Array3<u8>
            Input image (uint8).
        low_p (u8):
            Low percentile of image values to be saturated to 0.
        high_p (u8):
            High percentile of image values to be saturated to 255.
            Should be greater than low_p and at most 100.

    Returns:
        Array3<u8>
            Image with contrast enhanced.

    */
    let len = img.len();
    let mut flat_img_out: Vec<u8> = img.iter().copied().collect();
    flat_img_out.sort_unstable();

    let lenf32: f32 = len as f32;

    let p_low_index: f32 = (lenf32 - 1.0) * low_p as f32 / 100.0;
    let p_low_index_difference = p_low_index - p_low_index.floor();
    let mut p_low = (1.0 - p_low_index_difference)
        * flat_img_out[p_low_index.floor() as usize] as f32
        + p_low_index_difference * flat_img_out[p_low_index.ceil() as usize] as f32;

    let p_high_index: f32 = (lenf32 - 1.0) * high_p as f32 / 100.0;
    let p_high_index_difference = p_high_index - p_high_index.floor();
    let mut p_high = (1.0 - p_high_index_difference)
        * flat_img_out[p_high_index.floor() as usize] as f32
        + p_high_index_difference * flat_img_out[p_high_index.ceil() as usize] as f32;

    if p_low >= p_high {
        p_low = flat_img_out[0].into();
        p_high = flat_img_out[len - 1].into();
    }

    if p_high > p_low {
        let range = p_high - p_low;
        return img.mapv(|x| rescale_intensity(x.into(), p_low, p_high, range));
    }

    img
}

#[pyfunction]
fn contrast_enhancer<'py>(
    py: Python<'py>,
    img: PyReadonlyArray3<'py, u8>,
    low_p: u8,
    high_p: u8,
) -> Bound<'py, PyArray3<u8>> {
    let data = img.as_array().to_owned();
    rust_contrast_enhancer(data, low_p, high_p).into_pyarray(py)
}

/// Contrast enhancer implemented in Rust.
#[pymodule]
fn misc(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(contrast_enhancer, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests;
