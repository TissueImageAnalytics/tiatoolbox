use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
use pyo3::types::PyInt;
use pyo3::types::{PyDict, PyList};

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
    if class_name_is_none {
        single_qupath_feature.set_item("name", "object")?;
    } else {
        single_qupath_feature.set_item("name", &class_name)?;
    }
    Ok(single_qupath_feature.unbind())
}

#[pyfunction]
fn build_single_annotation(
    py: Python<'_>,
    np: &Bound<'_, PyAny>,
    i: i32,
    processed_predictions: &Bound<'_, PyDict>,
    class_dict: &Bound<'_, PyDict>,
) -> PyResult<Py<PyDict>> {
    let class_dict_is_none = class_dict.is_none();
    let properties = PyDict::new(py);
    let np_array = np.getattr("array")?;
    for (prop, arr) in processed_predictions.iter() {
        if prop.eq("type")? && !class_dict.len() == 0 {
            properties.set_item(prop, class_dict.get_item(arr.get_item(i)?)?)?;
        } else {
            properties.set_item(
                prop,
                np_array
                    .call1((arr.get_item(i)?,))?
                    .call_method0("tolist")?,
            )?;
        }
    }
    Ok(properties.unbind())
}

#[pyfunction]
fn compute_qupath_json(
    py: Python<'_>,
    class_dict: &Bound<'_, PyDict>,
    origin: (f64, f64),
    scale_factor: (f64, f64),
    batch_size: i32,
    verbose: bool,
    num_workers: i32,
    num_contours: i32,
    type_arr: &Bound<'_, PyList>,
    plt: &Bound<'_, PyAny>,
    py_build_single_qupath_feature: &Bound<'_, PyAny>,
    delayed: &Bound<'_, PyAny>,
    tqdm_dask_progress_bar: &Bound<'_, PyAny>,
) -> PyResult<Py<PyList>> {
    let builtins = py.import("builtins")?;
    let features = PyList::empty(py);
    if class_dict.len() == 0 {
        let valid_ids = PyList::empty(py);
        let mut valid_ids_contain_all_ints = true;
        for v in type_arr.iter() {
            if !v.is_none() {
                valid_ids.append(&v)?;
                if !v.is_instance_of::<PyInt>() {
                    valid_ids_contain_all_ints = false;
                }
            }
        }
        if valid_ids.len() == 0 {
            class_dict.set_item(0, 0)?;
        } else if valid_ids_contain_all_ints {
            let max_class: i64 = builtins.getattr("max")?.call1((&valid_ids,))?.extract()?;
            for i in 0..max_class + 1 {
                class_dict.set_item(i, i)?;
            }
        } else {
            let unique_names = builtins
                .getattr("sorted")?
                .call1((builtins.getattr("set")?.call1((&valid_ids,))?,))?;
            for name in unique_names.try_iter()? {
                let name = name?;
                class_dict.set_item(&name, &name)?;
            }
        }
    }
    let class_keys = class_dict.keys();
    let num_classes = class_keys.len();

    let colormaps = plt.getattr("colormaps")?;
    let tab20 = colormaps.get_item("tab20")?;
    let cmap = tab20.call_method1("resampled", (num_classes,))?;

    let class_colors = PyDict::new(py);

    for (i, key) in class_keys.iter().enumerate() {
        let rgba = cmap.call1((i,))?;

        let r: f64 = rgba.get_item(0)?.extract()?;
        let g: f64 = rgba.get_item(1)?.extract()?;
        let b: f64 = rgba.get_item(2)?.extract()?;

        let color = vec![(r * 255.0) as i64, (g * 255.0) as i64, (b * 255.0) as i64];

        class_colors.set_item(key, color)?;
    }
    for batch_id in (0..num_contours).step_by(batch_size as usize) {
        let delayed_tasks = PyList::empty(py);
        for i in batch_id..std::cmp::min(batch_id + batch_size, num_contours) {
            delayed_tasks.append(delayed.call1((py_build_single_qupath_feature,))?.call1((
                i,
                class_dict,
                origin,
                scale_factor,
                &class_colors,
            ))?)?;
        }
        let kwargs = PyDict::new(py);
        kwargs.set_item("write_tasks", delayed_tasks)?;
        kwargs.set_item("desc", "Computing QuPath features")?;
        kwargs.set_item("verbose", verbose)?;
        kwargs.set_item("num_workers", num_workers)?;
        let feature = tqdm_dask_progress_bar.call((), Some(&kwargs))?;
        for f in feature.try_iter()? {
            features.append(f?)?;
        }
    }
    Ok(features.unbind())
}

#[pyfunction]
fn compute_annotations(
    py: Python<'_>,
    store: &Bound<'_, PyAny>,
    class_dict: &Bound<'_, PyDict>,
    origin: (f64, f64),
    scale_factor: (f64, f64),
    batch_size: i32,
    num_workers: i32,
    verbose: bool,
    num_contours: i32,
    py_build_single_annotation: &Bound<'_, PyAny>,
    delayed: &Bound<'_, PyAny>,
    tqdm_dask_progress_bar: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    /*Compute annotations in batches and write them to a SQLiteStore.

    This method creates Dask Delayed tasks in batches to reduce scheduler
    overhead. Each batch is computed and written immediately using
    ``store.append_many()``.

    Args:
        store (SQLiteStore):
            A TIAToolbox SQLiteStore instance used to write annotations.

        class_dict (dict[int, str] | None):
            Optional mapping from integer class IDs to string labels.

        origin (tuple[float, float], optional):
            Translation offset ``(x, y)`` applied after scaling.
            Defaults to ``(0, 0)``.

        scale_factor (tuple[float, float], optional):
            Scaling factors ``(sx, sy)`` applied to contour coordinates.
            Defaults to ``(1, 1)``.

        batch_size (int, optional):
            Number of annotations to compute per batch. Larger batches
            reduce Dask scheduler overhead. Defaults to ``100``.

        num_workers (int, optional):
            Number of Dask workers to use. ``0`` means auto-detect.
            Passed through to the progress bar helper. Defaults to ``0``.

        verbose (bool, optional):
            Whether to display progress bars. Defaults to ``True``.

    Returns:
        SQLiteStore:
            The same store instance, after all annotations have been written.

    */
    for batch_id in (0..num_contours).step_by(batch_size as usize) {
        let delayed_tasks = PyList::empty(py);
        for i in batch_id..std::cmp::min(batch_id + batch_size, num_contours) {
            delayed_tasks.append(delayed.call1((py_build_single_annotation,))?.call1((
                i,
                class_dict,
                origin,
                scale_factor,
            ))?)?;
        }
        let kwargs = PyDict::new(py);
        kwargs.set_item("write_tasks", delayed_tasks)?;
        kwargs.set_item("desc", "Saving annotations")?;
        kwargs.set_item("verbose", verbose)?;
        kwargs.set_item("num_workers", num_workers)?;
        let feature = tqdm_dask_progress_bar.call((), Some(&kwargs))?;
        store.call_method1("append_many", (feature,))?;
    }
    Ok(store.into())
}

#[pymodule]
fn rmultitask(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(build_single_qupath_feature, m)?)?;
    m.add_function(wrap_pyfunction!(build_single_annotation, m)?)?;
    m.add_function(wrap_pyfunction!(compute_qupath_json, m)?)?;
    m.add_function(wrap_pyfunction!(compute_annotations, m)?)?;
    Ok(())
}
