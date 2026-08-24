use rmisc::add;

#[test]
fn test_add() {
    let result = add(2, 3);

    assert_eq!(result, 5);
}
