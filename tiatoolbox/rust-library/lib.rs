use ndarray::Axis;
use ndarray::{Array1, Array3};
use numpy::PyReadonlyArrayDyn;
use numpy::PyUntypedArrayMethods;
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray2, PyReadonlyArray3};
use ordered_float::OrderedFloat;
use pyo3::FromPyObject;
use pyo3::prelude::*;
use pyo3::pyclass::CompareOp;
use pyo3::types::{PyDict, PyList};
use pythonize::depythonize;
use serde_json::Value;
use std::collections::HashMap;

#[derive(FromPyObject)]
enum StringOrFloat {
    String(String),
    Float(f64),
}
#[pyfunction]
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

#[pyfunction]
pub fn string_to_tuple(in_str: String) -> Vec<String> {
    /*Splits input string to tuple at ','.

    Args:
        in_str (str):
            input string.

    */
    in_str
        .split(',')
        .map(|substring| substring.trim().to_string())
        .collect()
}

#[pyfunction]
pub fn semantic_segmentations_as_qupath_json(
    py: Python<'_>,
    layer_list: &Bound<'_, PyList>,
    preds: &Bound<'_, PyAny>,
    scale_factor: (f64, f64),
    class_dict: &Bound<'_, PyDict>,
    class_colours: &Bound<'_, PyDict>,
    cv2: &Bound<'_, PyAny>,
    poly_geo_fun: &Bound<'_, PyAny>,
) -> PyResult<Py<PyList>> {
    /*Helper function to save semantic segmentation as QuPath json.*/
    let class_colours: HashMap<OrderedFloat<f64>, Vec<i32>> = class_colours
        .iter()
        .map(|(key, value)| {
            let key: f64 = key.extract()?;
            let value: Vec<i32> = value.extract()?;

            Ok((OrderedFloat(key), value))
        })
        .collect::<PyResult<_>>()?;
    let features = PyList::empty(py);
    let retr_ccomp = cv2.getattr("RETR_CCOMP")?;
    let chain_approx_none = cv2.getattr("CHAIN_APPROX_NONE")?;
    let find_contours = cv2.getattr("findContours")?;
    for type_class in layer_list.iter() {
        let class_id: i64 = type_class.extract()?;
        let class_label = class_dict.get_item(class_id)?;
        let layer = preds
            .rich_compare(class_id, CompareOp::Eq)?
            .call_method1("astype", ("uint8",))?
            .call_method0("compute")?;
        let result = find_contours.call1((layer, retr_ccomp.clone(), chain_approx_none.clone()))?;

        let result = result.cast::<pyo3::types::PyTuple>()?;

        let contours = result.get_item(0)?;

        for cnt in contours.try_iter()? {
            let cnt = cnt?;
            let py_array = cnt.cast::<PyArray3<i32>>()?;
            //let array = py_array.to_owned();
            if py_array.shape()[0] >= 3 {
                let cnt_array: PyReadonlyArrayDyn<'_, i32> = cnt.extract()?;
                let cnt_scaled = cnt_array.as_array().index_axis_move(Axis(1), 0);
                let exterior: Vec<(f64, f64)> = cnt_scaled
                    .outer_iter()
                    .map(|p| (p[0] as f64 * scale_factor.0, p[1] as f64 * scale_factor.1))
                    .collect();
                let coordinates = vec![exterior];
                let poly_geo = poly_geo_fun.call1((coordinates,))?;
                let feature = PyDict::new(py);
                feature.set_item("type", "Feature")?;
                feature.set_item("geometry", poly_geo)?;
                feature.set_item("id", format!("class_{}_{}", class_id, features.len()))?;
                let classification = PyDict::new(py);
                classification.set_item("name", &class_label)?;
                classification.set_item(
                    "color",
                    class_colours[&OrderedFloat(class_id as f64)].clone(),
                )?;
                let properties = PyDict::new(py);
                properties.set_item("classification", classification)?;
                feature.set_item("properties", properties)?;
                feature.set_item("objectType", "annotation")?;
                feature.set_item("name", &class_label)?;
                feature.set_item("class_value", class_id)?;
                features.append(feature)?;
            }
        }
    }
    Ok(features.unbind())
}

#[pyfunction]
fn json_dump_python_object(save_path: String, obj: &Bound<'_, PyAny>) -> PyResult<()> {
    //Equilivent to json.dump(obj, save_path)
    //Caution: if obj is a dictionary and has a key of an integer it will throw an error
    let value: Value =
        depythonize(obj).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let file = std::fs::File::create(&save_path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    let mut writer = std::io::BufWriter::new(file);

    serde_json::to_writer_pretty(&mut writer, &value)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    /*
    serde_json::to_writer(&mut writer, &value)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    */
    Ok(())
}

#[pyfunction]
fn patch_predictions_as_annotations<'py>(
    py: Python<'_>,
    annotation_class: &Bound<'_, PyAny>,
    polygon_class: &Bound<'_, PyAny>,
    preds: Vec<f64>,
    keys_contains_labels: bool,
    keys_contains_probabilities: bool,
    class_dict: &Bound<'_, PyDict>,
    py_class_probs: PyReadonlyArray2<'py, f64>,
    py_patch_coords: PyReadonlyArray2<'py, f64>,
    classes_predicted: Vec<f64>,
    labels: Vec<f64>,
) -> PyResult<Vec<Py<PyAny>>> {
    /*Helper function to generate annotation per patch predictions.*/
    let class_dict: HashMap<OrderedFloat<f64>, StringOrFloat> = class_dict
        .iter()
        .map(|(key, value)| {
            let key: f64 = key.extract()?;
            let value: StringOrFloat = value.extract()?;
            Ok((OrderedFloat(key), value))
        })
        .collect::<PyResult<_>>()?;
    let class_probs = py_class_probs.as_array();
    let patch_coords = py_patch_coords.as_array();
    let mut annotations: Vec<Py<PyAny>> = Vec::with_capacity(patch_coords.nrows());
    let preds_len = preds.len();
    for i in 0..patch_coords.nrows() {
        let props = PyDict::new(py);
        if keys_contains_probabilities {
            for j in &classes_predicted {
                let y = &class_dict[&OrderedFloat(*j)];
                let probability = match y {
                    StringOrFloat::String(s) => s.clone(),
                    StringOrFloat::Float(i) => i.to_string(),
                };
                props.set_item(
                    format!("prob_{}", probability),
                    class_probs[[i, *j as usize]],
                )?;
            }
        }
        if keys_contains_labels {
            let y = &class_dict[&OrderedFloat(labels[i])];
            match y {
                StringOrFloat::String(s) => {
                    props.set_item("label".to_string(), s)?;
                }
                StringOrFloat::Float(i) => {
                    props.set_item("label".to_string(), *i)?;
                }
            }
        }
        if preds_len > 0 {
            let y = &class_dict[&OrderedFloat(preds[i])];
            match y {
                StringOrFloat::String(s) => {
                    props.set_item("type".to_string(), s)?;
                }
                StringOrFloat::Float(i) => {
                    props.set_item("type".to_string(), *i)?;
                }
            }
        }
        annotations.push(
            annotation_class
                .call1((
                    polygon_class.call_method1(
                        "from_bounds",
                        (
                            patch_coords[[i, 0]],
                            patch_coords[[i, 1]],
                            patch_coords[[i, 2]],
                            patch_coords[[i, 3]],
                        ),
                    )?,
                    props,
                ))?
                .unbind(),
        );
    }
    Ok(annotations)
}

#[pyfunction]
fn patch_predictions_as_qupath_json<'py>(
    py: Python<'_>,
    class_colours: &Bound<'_, PyDict>,
    preds: Vec<f64>,
    class_dict: &Bound<'_, PyDict>,
    py_patch_coords: PyReadonlyArray2<'py, f64>,
) -> PyResult<Py<PyList>> {
    /*Helper function to generate QuPath JSON per patch predictions.*/
    let class_colours: HashMap<OrderedFloat<f64>, Vec<i32>> = class_colours
        .iter()
        .map(|(key, value)| {
            let key: f64 = key.extract()?;
            let value: Vec<i32> = value.extract()?;

            Ok((OrderedFloat(key), value))
        })
        .collect::<PyResult<_>>()?;
    let class_dict: HashMap<OrderedFloat<f64>, String> = class_dict
        .iter()
        .map(|(key, value)| {
            let key: f64 = key.extract()?;
            let value: String = value.extract::<String>()?;

            Ok((OrderedFloat(key), value))
        })
        .collect::<PyResult<_>>()?;
    let features = PyList::empty(py);
    let patch_coords = py_patch_coords.as_array();
    for i in 0..patch_coords.nrows() {
        let class_idx = preds[i];
        let class_name = &class_dict[&OrderedFloat(class_idx)];
        let xmin = patch_coords[[i, 0]];
        let ymin = patch_coords[[i, 1]];
        let xmax = patch_coords[[i, 2]];
        let ymax = patch_coords[[i, 3]];
        let polygon_feat = PyDict::new(py);
        polygon_feat.set_item("type", "Polygon")?;
        polygon_feat.set_item(
            "coordinates",
            ((
                (xmin, ymin),
                (xmin, ymax),
                (xmax, ymax),
                (xmax, ymin),
                (xmin, ymin),
            ),),
        )?;
        let feature = PyDict::new(py);
        feature.set_item("type", "Feature")?;
        feature.set_item("id", format!("patch_{}", i))?;
        feature.set_item("geometry", polygon_feat)?;
        let classification = PyDict::new(py);
        classification.set_item("name", class_name)?;
        classification.set_item("color", class_colours[&OrderedFloat(class_idx)].clone())?;
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
    let mut flat_img_out: Array1<u8> = img_out.into_shape_with_order(len).unwrap();

    if let Some(slice) = flat_img_out.as_slice_mut() {
        slice.sort_unstable();
    }

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
pub fn contrast_enhancer<'py>(
    py: Python<'py>,
    img: PyReadonlyArray3<'py, u8>,
    low_p: u8,
    high_p: u8,
) -> Bound<'py, PyArray3<u8>> {
    let data = img.as_array().to_owned();
    rust_contrast_enhancer(data, low_p, high_p).into_pyarray(py)
}

#[pymodule]
fn rmisc(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(contrast_enhancer, m)?)?;
    m.add_function(wrap_pyfunction!(patch_predictions_as_qupath_json, m)?)?;
    m.add_function(wrap_pyfunction!(patch_predictions_as_annotations, m)?)?;
    m.add_function(wrap_pyfunction!(json_dump_python_object, m)?)?;
    m.add_function(wrap_pyfunction!(string_to_tuple, m)?)?;
    m.add_function(wrap_pyfunction!(semantic_segmentations_as_qupath_json, m)?)?;
    Ok(())
}
