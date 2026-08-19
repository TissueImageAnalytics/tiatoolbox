use pyo3::prelude::*;
use ndarray::{Array1, Array3};
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray3};

#[pyfunction]
fn add(a: i32, b: i32) -> i32 {
    a + b
}

fn rescale_intensity(x: f32, in_range_low: f32, in_range_high: f32, range: f32) -> u8{
    //asssumes out_min = 0 and out_max = 255
    if x <= in_range_low {
        0
    } else if x >= in_range_high {
        255
    } else {
        (255.0 * ((x-in_range_low)/range)) as u8
    }
}

fn rust_contrast_enhancer(img: Array3<u8>, low_p: u8, high_p: u8) -> Array3<u8> {
    /*Get tissue mask based on the luminosity of the input image.

    Args:
        img: Array
            Input image used to obtain tissue mask.
        threshold (float):
            Luminosity threshold used to determine tissue area.

    Returns:
        tissue_mask
            Binary tissue mask.

    */
    let img_out = img.to_owned();
    let len = img.len();
    let mut flat_img_out: Array1<u8> = img_out.into_shape_with_order(len, ).unwrap();

    if let Some(slice) = flat_img_out.as_slice_mut() {
        slice.sort_unstable();
    }

    let lenf32: f32 = len as f32;

    let p_low_index: f32 = (lenf32 - 1.0) * low_p as f32 / 100.0;
    let p_low_index_difference = p_low_index - p_low_index.floor();
    let mut p_low = (1.0 - p_low_index_difference) * flat_img_out[p_low_index.floor() as usize] as f32
                    + p_low_index_difference * flat_img_out[p_low_index.ceil() as usize] as f32;

    let p_high_index: f32 = (lenf32 - 1.0) * high_p as f32 / 100.0;
    let p_high_index_difference = p_high_index - p_high_index.floor();
    let mut p_high = (1.0 - p_high_index_difference) * flat_img_out[p_high_index.floor() as usize] as f32
                    + p_high_index_difference * flat_img_out[p_high_index.ceil() as usize] as f32;

    if p_low >= p_high {
        p_low = flat_img_out[0].into();
        p_high = flat_img_out[len - 1].into();
    }

    if p_high > p_low {
        let range = p_high - p_low ;
        return img.mapv(|x| rescale_intensity(x.into(), p_low, p_high, range));
    }

    return img
}

#[pyfunction]
fn contrast_enhancer<'py>(py: Python<'py>, img: PyReadonlyArray3<'py, u8>, low_p: u8, high_p: u8) -> Bound<'py, PyArray3<u8>> {
    let data = img.as_array().to_owned();
    rust_contrast_enhancer(data, low_p, high_p).into_pyarray(py)
}

#[pymodule]
fn miscrust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(contrast_enhancer, m)?)?;

    Ok(())
}
