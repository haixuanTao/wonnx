#![cfg(test)]

use std::collections::HashMap;

use protobuf::ProtobufEnum;
use wonnx::{
    onnx::TensorProto_DataType,
    onnx_model::{
        onnx_attribute, onnx_graph, onnx_model, onnx_node, onnx_tensor, onnx_tensor_of_type,
    },
    tensor::TensorData,
};

#[test]
fn test_cast() {
    let n: usize = 16;
    let mut input_data = HashMap::new();

    // The input will be 0.0, 0.33, 0.66, ... and (by casting) we'll effectively 'floor' these
    let data: Vec<f32> = (0..n).map(|x| x as f32 / 3.0).collect();
    let dims = vec![n as i64];
    input_data.insert("X".to_string(), data.as_slice().into());

    // Model: X -> Identity -> Y; Y==Z
    let model = onnx_model(onnx_graph(
        vec![onnx_tensor("X", &dims)],
        vec![onnx_tensor_of_type("Y", &dims, TensorProto_DataType::INT32)],
        vec![],
        vec![],
        vec![onnx_node(
            vec!["X"],
            vec!["Y"],
            "Cast",
            vec![onnx_attribute(
                "to",
                TensorProto_DataType::INT32.value() as i64,
            )],
        )],
    ));

    let session =
        pollster::block_on(wonnx::Session::from_model(model)).expect("Session did not create");

    let result = pollster::block_on(session.run(&input_data)).unwrap();
    assert_eq!(
        result["Y"],
        TensorData::I32(vec![0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5].into())
    );
}
