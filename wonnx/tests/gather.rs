use std::{collections::HashMap, convert::TryInto};
use wonnx::onnx_model::{onnx_attribute, onnx_graph, onnx_model, onnx_node, onnx_tensor};
mod common;

fn assert_gather(
    data: &[f32],
    data_shape: &[i64],
    indices: &[i32],
    indices_shape: &[i64],
    output: &[f32],
    output_shape: &[i64],
    axis: i64,
) {
    let mut input_data = HashMap::new();

    input_data.insert("X".to_string(), data.into());
    input_data.insert("I".to_string(), indices.into());

    // Model: (X, I) -> Gather -> Y
    let bn_model = onnx_model(onnx_graph(
        vec![
            onnx_tensor("X", data_shape),
            onnx_tensor("I", indices_shape),
        ],
        vec![onnx_tensor("Y", output_shape)],
        vec![],
        vec![],
        vec![onnx_node(
            vec!["X", "I"],
            vec!["Y"],
            "Gather",
            vec![onnx_attribute("axis", axis)],
        )],
    ));

    let session =
        pollster::block_on(wonnx::Session::from_model(bn_model)).expect("Session did not create");

    let result = pollster::block_on(session.run(&input_data)).unwrap();
    common::assert_eq_vector((&result["Y"]).try_into().unwrap(), output);
}

#[test]
fn gather() {
    let _ = env_logger::builder().is_test(true).try_init();

    // Very simple test case that just does simple selection from a 1D array
    assert_gather(
        &[1.0, 1.2, 2.3, 3.4, 4.5, 5.7],
        &[6],
        &[3, 2, 3, 1],
        &[4],
        &[3.4, 2.3, 3.4, 1.2],
        &[4],
        0,
    );

    // Very simple test case that just does simple selection from a 1D array, with negative indexing
    assert_gather(
        &[1.0, 1.2, 2.3, 3.4, 4.5, 5.7],
        &[6],
        &[-3, -2, -3, -1],
        &[4],
        &[3.4, 4.5, 3.4, 5.7],
        &[4],
        0,
    );

    // Test case for axis=0 from https://github.com/onnx/onnx/blob/main/docs/Operators.md#gather
    assert_gather(
        &[1.0, 1.2, 2.3, 3.4, 4.5, 5.7],
        &[3, 2],
        &[0, 1, 1, 2],
        &[2, 2],
        &[1.0, 1.2, 2.3, 3.4, 2.3, 3.4, 4.5, 5.7],
        &[2, 2, 2],
        0,
    );

    // Same as the above, but now with larger chunks to copy (so we test the shader's batching capability)
    assert_gather(
        &[1.0, 1.2, 2.3, 3.4, 4.5, 5.7, 1.0, 1.2, 2.3, 3.4, 4.5, 5.7],
        &[3, 4],
        &[0, 1, 1, 2],
        &[2, 2],
        &[
            1.0, 1.2, 2.3, 3.4, 4.5, 5.7, 1.0, 1.2, 4.5, 5.7, 1.0, 1.2, 2.3, 3.4, 4.5, 5.7,
        ],
        &[2, 2, 4],
        0,
    );
}
