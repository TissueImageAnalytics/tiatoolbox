use rmisc::{add, string_to_tuple};
use std::process::{Command, Stdio};

#[test]
fn test_add() {
    let result = add(2, 3);

    assert_eq!(result, 5);
}

#[test]
fn test_string_to_tuple() {
    let in_str = "a, b, c".to_string();
    let result = string_to_tuple(in_str);

    assert_eq!(result, vec!["a", "b", "c"])
}
'''
#[test]
fn test_misc() {
    let status = Command::new("python3")
        .args(["-m", "pytest", "tests/test_rust.py", "-v"])
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .expect("Failed to run pytest");

    if status.success() {
        println!("Tests passed!");
    } else {
        eprintln!("Tests failed!");
        std::process::exit(1);
    }
}
'''
