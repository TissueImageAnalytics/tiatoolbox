use pyo3::prelude::*;
use ndarray::{Array1, Array3};
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::types::{PyList, PyDict};
use std::collections::HashMap;
use pythonize::depythonize;
use serde_json::Value;
use ordered_float::OrderedFloat;

#[pyfunction]
fn add(a: i32, b: i32) -> i32 {
    a + b
}

#[pyfunction]
fn json_dump_python_object(save_path: String, obj: &Bound<'_, PyAny>) -> PyResult<()> {
    //Equilivent to json.dump(obj, save_path)
    let value: Value = depythonize(obj)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let file = std::fs::File::create(&save_path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    let mut writer = std::io::BufWriter::new(file);

    serde_json::to_writer(&mut writer, &value)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    Ok(())
}

#[pyfunction]
fn patch_predictions_as_annotations<'py>(
        py: Python<'_>,
        annotation_class: &Bound<'_, PyAny>,
        polygon_class: &Bound<'_, PyAny>,
        preds: Vec<f64>,
        keys: Vec<String>,
        class_dict: &Bound<'_, PyDict>,
        py_class_probs: PyReadonlyArray2<'py, f64>,
        py_patch_coords: PyReadonlyArray2<'py, f64>,
        classes_predicted: Vec<i32>,
        labels: Vec<f64>
    ) -> PyResult<Vec<Py<PyAny>>>{
    /*Helper function to generate annotation per patch predictions.*/
    let class_dict: HashMap<OrderedFloat<f64>, String> = class_dict
        .iter()
        .map(|(key, value)| {
            let key: f64 = key.extract()?;
            let value: String = value.extract()?;

            Ok((OrderedFloat(key), value))
        })
        .collect::<PyResult<_>>()?;
    let class_probs = py_class_probs.as_array();
    let patch_coords = py_patch_coords.as_array();
    let mut annotations: Vec<Py<PyAny>>  = Vec::with_capacity(patch_coords.nrows());
    let preds_len = preds.len();
    let keys_contains_labels = keys.contains(&"labels".to_string());
    for i in 0..patch_coords.nrows() {
        let props = PyDict::new(py);
        if keys.contains(&"probabilities".to_string()) {
            for j in &classes_predicted {
                props.set_item(format!("prob_{}", class_dict[&OrderedFloat(*j as f64)]), class_probs[[i, *j as usize]])?;
            }
        }
        if keys_contains_labels {
            props.set_item("label".to_string(), class_dict[&OrderedFloat(labels[i])].clone())?;
        }
        if preds_len > 0 {
            props.set_item("type".to_string(), class_dict[&OrderedFloat(preds[i])].clone())?;
        }
        annotations.push(annotation_class.call1((
            polygon_class.call_method1("from_bounds", (
            patch_coords[[i, 0]],
            patch_coords[[i, 1]],
            patch_coords[[i, 2]],
            patch_coords[[i, 3]]
        ))?,
        props))?.unbind());
    }
    Ok(annotations)
}

#[pyfunction]
fn patch_predictions_as_qupath_json<'py>(py: Python<'_>,
        class_colours: HashMap<i32, Vec<i32>>,
        preds: Vec<i32>,
        class_dict: HashMap<i32, String>,
        py_patch_coords: PyReadonlyArray2<'py, f64>)
        -> PyResult<Py<PyList>> {
    /*Helper function to generate QuPath JSON per patch predictions.*/

    let features = PyList::empty(py);
    let patch_coords = py_patch_coords.as_array();
    for i in 0..patch_coords.nrows() {
        let class_idx = preds[i];
        let class_name = &class_dict[&class_idx];
        let xmin = patch_coords[[i, 0]];
        let ymin = patch_coords[[i, 1]];
        let xmax = patch_coords[[i, 2]];
        let ymax = patch_coords[[i, 3]];
        let polygon_feat = PyDict::new(py);
        polygon_feat.set_item("type", "Polygon")?;
        polygon_feat.set_item(
            "coordinates",
            vec![vec![
                [xmin, ymin],
                [xmin, ymax],
                [xmax, ymax],
                [xmax, ymin],
                [xmin, ymin],
            ]],
        )?;
        let feature = PyDict::new(py);
        feature.set_item("type", "Feature")?;
        feature.set_item("id", format!("patch_{}", i))?;
        feature.set_item("geometry", polygon_feat)?;
        let classification = PyDict::new(py);
        classification.set_item("name", class_name)?;
        classification.set_item("color", class_colours[&class_idx].clone())?;
        let properties = PyDict::new(py);
        properties.set_item("classification", classification)?;
        feature.set_item("properties", properties)?;
        feature.set_item("objectType", "annotation")?;
        feature.set_item("name", class_name)?;
        feature.set_item("class_value", class_idx)?;
        features.append(feature)?;
    }

    Ok(features.unbind())
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
    m.add_function(wrap_pyfunction!(patch_predictions_as_qupath_json, m)?)?;
    m.add_function(wrap_pyfunction!(patch_predictions_as_annotations, m)?)?;
    m.add_function(wrap_pyfunction!(json_dump_python_object, m)?)?;
    Ok(())
}
