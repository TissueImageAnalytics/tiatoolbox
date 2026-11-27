use super::*;

#[test]
fn rescale_intensity_saturates_and_scales() {
    assert_eq!(rescale_intensity(5.0, 10.0, 20.0, 10.0), 0);
    assert_eq!(rescale_intensity(10.0, 10.0, 20.0, 10.0), 0);
    assert_eq!(rescale_intensity(20.0, 10.0, 20.0, 10.0), 255);
    assert_eq!(rescale_intensity(25.0, 10.0, 20.0, 10.0), 255);
    assert_eq!(rescale_intensity(15.0, 10.0, 20.0, 10.0), 127);
}

#[test]
fn contrast_enhancer_matches_python_reference() {
    let input = Array3::from_shape_vec(
        (1, 3, 9),
        vec![
            37, 244, 193, 106, 235, 128, 71, 140, 47, //
            103, 184, 72, 20, 188, 238, 126, 7, 0, //
            137, 195, 204, 32, 203, 170, 101, 77, 133,
        ],
    )
    .unwrap();
    let expected = Array3::from_shape_vec(
        (1, 3, 9),
        vec![
            35, 255, 203, 110, 248, 133, 72, 146, 46, //
            106, 193, 73, 17, 198, 251, 131, 3, 0, //
            143, 205, 215, 30, 214, 178, 104, 78, 139,
        ],
    )
    .unwrap();
    assert_eq!(rust_contrast_enhancer(input, 2, 98), expected);
}

#[test]
fn contrast_enhancer_full_range_stretch() {
    let input = Array3::from_shape_vec((1, 1, 3), vec![10, 20, 30]).unwrap();
    let expected = Array3::from_shape_vec((1, 1, 3), vec![0, 127, 255]).unwrap();
    assert_eq!(rust_contrast_enhancer(input, 0, 100), expected);
}

#[test]
fn contrast_enhancer_falls_back_to_min_max() {
    let mut input = Array3::<u8>::zeros((20, 20, 3));
    input[[0, 0, 0]] = 255;
    assert_eq!(rust_contrast_enhancer(input.clone(), 2, 98), input);
}

#[test]
fn contrast_enhancer_constant_image_unchanged() {
    let input = Array3::from_elem((4, 4, 3), 77u8);
    assert_eq!(rust_contrast_enhancer(input.clone(), 2, 98), input);
}
