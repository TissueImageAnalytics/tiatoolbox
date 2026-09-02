//Tests that functions in rust are working as expected

use misc::string_to_tuple;

#[test]
fn test_string_to_tuple() {
    /*Tests function string to tuple*/
    let in_str = "a, b, c".to_string();
    let result = string_to_tuple(in_str);

    assert_eq!(result, vec!["a", "b", "c"])
}
