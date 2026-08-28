use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::IntoPyObjectExt;

#[derive(Clone)]
enum StringOrFloat {
    String(String),
    Float(f64),
}

#[pyfunction]
fn add(a: i32, b: i32) -> i32 {
    a + b
}

#[pyfunction]
fn build_single_qupath_feature<'py>(
    py: pyo3::Python<'py>,
    geo_map: &Bound<'_, PyAny>,
    processed_predictions: &pyo3::Bound<'py, PyDict>,
    i: i32,
    class_dict: &pyo3::Bound<'py, PyDict>,
    class_colours: &Bound<'_, PyDict>,
) -> PyResult<Py<PyDict>> {
    let props = PyDict::new(py);
    let mut class_value: Option<Bound<'py, PyAny>> = None;
    let mut class_name: Option<Bound<'py, PyAny>> = None;
    for (key, arr) in processed_predictions.iter() {
        let item = arr.get_item(i)?;
        let value = if item.hasattr("tolist")? {
            item.call_method0("tolist")?
        } else {
            item
        };
        if key.eq("type")? {
            if value.is_none() {
                class_value = Some(0_i64.into_bound_py_any(py)?);
                class_name = match class_dict.get_item(0)? {
                    Some(v) => Some(v),
                    None => Some(0_i64.into_bound_py_any(py)?),
                };
                props.set_item("type", &class_name)?;
            } else {
                if !class_dict.is_none() && class_dict.contains(&value)? {
                    if let Some(item) = class_dict.get_item(&value)? {
                        class_name = Some(item.clone());
                    }
                } else {
                    class_name = Some(value.clone());
                }
                props.set_item("type", &class_name)?;
                class_value = Some(value.clone());
            }
        } else if !value.is_none() {
            props.set_item(key, &value)?;
        }
    }
    let class_name_is_none = match class_name {
        Some(ref _s) => false,
        None => true,
    };
    if !class_name_is_none && class_colours.contains(&class_value)? {
        let color = class_colours.get_item(&class_value)?;
        let classification_dict = PyDict::new(py);
        classification_dict.set_item("name", &class_name)?;
        classification_dict.set_item("color", color)?;
        props.set_item("classification", classification_dict)?;
        props.set_item("class_value", class_value)?;
    }
    let single_qupath_feature = PyDict::new(py);
    single_qupath_feature.set_item("type", "Feature")?;
    single_qupath_feature.set_item("id", format!("object_{}", i))?;
    single_qupath_feature.set_item("geometry", geo_map)?;
    single_qupath_feature.set_item("properties", props)?;
    single_qupath_feature.set_item("objectType", "annotation")?;
    if class_name.is_none() {
        single_qupath_feature.set_item("name", "object")?;
    } else {
        single_qupath_feature.set_item("name", &class_name)?;
    }
    Ok(single_qupath_feature.unbind())
}

#[pymodule]
fn rmultitask(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(build_single_qupath_feature, m)?)?;
    Ok(())
}
