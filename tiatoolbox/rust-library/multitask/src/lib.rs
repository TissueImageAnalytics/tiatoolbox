use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::ffi::PyObject;

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
fn build_single_qupath_feature(
    py: Python<'_>,
    np: &Bound<'_, PyAny>,
    geo_map: &Bound<'_, PyAny>,
    processed_predictions: &Bound<'_, PyDict>,
    i: i32,
    class_dict: &Bound<'_, PyDict>,
    class_colours: &Bound<'_, PyDict>,
) -> PyResult<Py<PyDict>> {
    let props = PyDict::new(py);
    let mut class_value: Option<StringOrFloat> = None;
    let mut class_name: Option<StringOrFloat> = None;
    for (key, arr) in processed_predictions.iter() {
        let item = arr.get_item(i)?;
        let value = if item.hasattr("tolist")? {
            item.call_method0("tolist")?
        } else {
            item
        };
        if key.eq("type")? {
            if value.is_none() {
                class_value = Some(StringOrFloat::Float(0.0));
                class_name = match class_dict.get_item(0)? {
                    Some(value) => {
                        if let Ok(s) = value.extract::<String>() {
                                Some(StringOrFloat::String(s))
                            } else if let Ok(f) = value.extract::<f64>() {
                                Some(StringOrFloat::Float(f))
                            } else {
                                None
                            }
                    },
                    None => Some(StringOrFloat::Float(0.0)),
                };
                match class_name.as_ref() {
                    Some(StringOrFloat::String(s)) => {
                        props.set_item("type", s)?;
                    }
                    Some(StringOrFloat::Float(i)) => {
                        props.set_item("type", i)?;
                    }
                    None => {
                        props.set_item("type", py.None())?;
                    }
                }
            } else {
                if !class_dict.is_none() && class_dict.contains(&value)? {
                    if let Some(item) = class_dict.get_item(&value)? {
                        class_name = if let Ok(s) = item.extract::<String>() {
                                Some(StringOrFloat::String(s))
                            } else if let Ok(f) = item.extract::<f64>() {
                                Some(StringOrFloat::Float(f))
                            } else {
                                None
                            }
                    }
                } else {
                    class_name = if let Ok(s) = value.extract::<String>() {
                                Some(StringOrFloat::String(s))
                            } else if let Ok(f) = value.extract::<f64>() {
                                Some(StringOrFloat::Float(f))
                            } else {
                                None
                            }
                }
                match class_name.as_ref() {
                    Some(StringOrFloat::String(s)) => {
                        props.set_item("type", s)?;
                    }
                    Some(StringOrFloat::Float(i)) => {
                        props.set_item("type", i)?;
                    }
                    None => {
                        props.set_item("type", py.None())?;
                    }
                }
                class_value = if let Ok(s) = value.extract::<String>() {
                                Some(StringOrFloat::String(s))
                            } else if let Ok(f) = value.extract::<f64>() {
                                Some(StringOrFloat::Float(f))
                            } else {
                                None
                            };
            }
        } else if !value.is_none() {
            props.set_item(
                key,
                np.call_method1("array", (value,))?.call_method0("tolist")?,
            )?;
        }
    }
    let class_colours_contains_class_value = match class_value {
        Some(StringOrFloat::String(ref s)) => class_colours.contains(&s)?,
        Some(StringOrFloat::Float(i)) => class_colours.contains(&i)?,
        None => false,
    };
    let class_name_is_none = match class_name {
        Some(ref _s) => false,
        None => true,
    };
    if !class_name_is_none && class_colours_contains_class_value {
        let color = match class_value {
            Some(StringOrFloat::String(ref s)) => class_colours.get_item(&s)?,
            Some(StringOrFloat::Float(i)) => class_colours.get_item(&i)?,
            None => None,
        };
        let classification_dict = PyDict::new(py);
        match class_name.as_ref() {
            Some(StringOrFloat::String(s)) => {
                classification_dict.set_item("name", s)?;
            }
            Some(StringOrFloat::Float(i)) => {
                classification_dict.set_item("name", i)?;
            }
            None => {
                classification_dict.set_item("name", py.None())?;
            }
        }
        classification_dict.set_item("color", color)?;
        props.set_item("classification", classification_dict)?;
        match class_value.as_ref() {
            Some(StringOrFloat::String(s)) => {
                props.set_item("class_value", s)?;
            }
            Some(StringOrFloat::Float(i)) => {
                props.set_item("class_value", i)?;
            }
            None => {
                props.set_item("class_value", py.None())?;
            }
        }
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
        match class_name.as_ref() {
            Some(StringOrFloat::String(s)) => {
                single_qupath_feature.set_item("name", s)?;
            }
            Some(StringOrFloat::Float(i)) => {
                single_qupath_feature.set_item("name", i)?;
            }
            None => {
                single_qupath_feature.set_item("name", py.None())?;
            }
        }
    }
    Ok(single_qupath_feature.unbind())
}

#[pymodule]
fn rmultitask(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(build_single_qupath_feature, m)?)?;
    Ok(())
}
