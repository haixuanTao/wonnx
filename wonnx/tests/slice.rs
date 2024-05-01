use std::{collections::HashMap, convert::TryInto, vec};
use wonnx::{
    onnx::AttributeProto,
    utils::{ graph, model, node, tensor, InputTensor},
};
mod common;

fn test_slice(
    input: &[f32],
    input_shape: &[i64],
    starts: &[i32],
    ends: &[i32],
    axes: Option<&[i64]>,
    steps: Option<&[i64]>,
    output: &[f32],
    output_shape: &[i64],
) {
    let mut input_data = HashMap::<String, InputTensor>::new();
    let mut inputs = vec![];
    let mut labels = vec![];

    input_data.insert("X".to_string(), input.into());
    inputs.push(tensor("X", input_shape));
    labels.push("X");

    input_data.insert("starts".to_string(), starts.into());
    inputs.push(tensor("starts", &[starts.len() as i64]));
    labels.push("starts");

    input_data.insert("ends".to_string(), ends.into());
    inputs.push(tensor("ends", &[ends.len() as i64]));
    labels.push("ends");

    if let Some(axes) = axes {
        input_data.insert("axes".to_string(), axes.into());
        inputs.push(tensor("axes", &[axes.len() as i64]));
        labels.push("axes");
    } 

    if let Some(steps) = steps {
        input_data.insert("steps".to_string(), steps.into());
        inputs.push(tensor("steps", &[steps.len() as i64]));
        labels.push("steps");
    }

    let attributes: Vec<AttributeProto> = vec![];

    let model = model(graph(
        inputs,
        vec![tensor("Y", output_shape)],
        vec![],
        vec![],
        vec![node(
            labels,
            vec!["Y"],
            "slice",
            "Slice",
            attributes,
        )],
    ));

    let session = match pollster::block_on(wonnx::Session::from_model(model)) {
        Ok(session) => session,
        Err(e) => {
            panic!("Failed to create session: {:?}", e);
        }
    };
    let result = pollster::block_on(session.run(&input_data)).unwrap();
    log::info!("OUT: {:?}", result["Y"]);
    common::assert_eq_vector((&result["Y"]).try_into().unwrap(), output);
}

#[test]
fn slice_step1() {
    let _ = env_logger::builder().is_test(true).try_init();

    // This test is the most simple case
    // Note that each interval is half-open, i.e. it includes the start index but excludes the end index.
    // axes == 0
    test_slice(
        &[[1., 2., 3., 4.], [5., 6., 7., 8.]].concat(),
        &[1, 2, 4],
        &[0],
        &[1],
        Some(&[0]),
        Some(&[1]),
        &[[1., 2., 3., 4.], [5., 6., 7., 8.]].concat(),
        &[1, 2, 4],
    );

    // axes == 1
    test_slice(
        &[[1., 2., 3., 4.], [5., 6., 7., 8.]].concat(),
        &[1, 2, 4],
        &[0],
        &[2],
        Some(&[1]),
        Some(&[1]),
        &[[1., 2., 3., 4.], [5., 6., 7., 8.]].concat(),
        &[1, 2, 4],
    );

    // axes == 2
    test_slice(
        &[[1., 2., 3., 4.], [5., 6., 7., 8.]].concat(),
        &[1, 2, 4],
        &[0],
        &[4],
        Some(&[2]),
        Some(&[1]),
        &[[1., 2., 3., 4.], [5., 6., 7., 8.]].concat(),
        &[1, 2, 4],
    );

    // axes == 1
    test_slice(
        &[[1., 2., 3., 4.], [5., 6., 7., 8.]].concat(),
        &[1, 2, 4],
        &[0],
        &[1],
        Some(&[1]),
        Some(&[1]),
        &[[1., 2., 3., 4.]].concat(),
        &[1, 1, 4],
    );

    // axes == 1
    test_slice(
        &[[1., 2., 3., 4.], [5., 6., 7., 8.]].concat(),
        &[1, 2, 4],
        &[1],
        &[2],
        Some(&[1]),
        Some(&[1]),
        &[[5., 6., 7., 8.]].concat(),
        &[1, 1, 4],
    );

    // axes == 2
    test_slice(
        &[[1., 2., 3., 4.], [5., 6., 7., 8.]].concat(),
        &[1, 2, 4],
        &[0],
        &[2],
        Some(&[2]),
        Some(&[1]),
        &[[1., 2.], [5., 6.]].concat(),
        &[1, 2, 2],
    );

    // axes == 2
    test_slice(
        &[[1., 2., 3., 4.], [5., 6., 7., 8.]].concat(),
        &[1, 2, 4],
        &[2],
        &[4],
        Some(&[2]),
        Some(&[1]),
        &[[3., 4.], [7., 8.]].concat(),
        &[1, 2, 2],
    );
}

#[test]
fn slice_step2() {
    // axes == 0
    #[rustfmt::skip]    
    test_slice(
        &[
            [ 1.,  2.,  3.,  4.], 
            [ 5.,  6.,  7.,  8.],
            [ 9., 10., 11., 12.], 
            [13., 14., 15., 16.]
        ].concat(),
        &[1, 4, 4, 1],
        &[0],
        &[1],
        Some(&[0]),
        Some(&[2]),
        &[
            [ 1.,  2.,  3.,  4.], 
            [ 5.,  6.,  7.,  8.],
            [ 9., 10., 11., 12.], 
            [13., 14., 15., 16.]
        ].concat(),
        &[1, 4, 4, 1],
    );

    #[rustfmt::skip]    
    test_slice(
        &[
            [ 1.,  2.,  3.,  4.], 
            [ 5.,  6.,  7.,  8.],
            [ 9., 10., 11., 12.], 
            [13., 14., 15., 16.]
        ].concat(),
        &[1, 4, 4, 1],
        &[0],
        &[4],
        Some(&[1]),
        Some(&[2]),
        &[
            [ 1.,  2.,  3.,  4.], 
            [ 9., 10., 11., 12.], 
        ].concat(),
        &[1, 2, 4, 1],
    );

    #[rustfmt::skip]    
    test_slice(
        &[
            [ 1.,  2.,  3.,  4.], 
            [ 5.,  6.,  7.,  8.],
            [ 9., 10., 11., 12.], 
            [13., 14., 15., 16.]
        ].concat(),
        &[1, 4, 4, 1],
        &[1],
        &[4],
        Some(&[1]),
        Some(&[2]),
        &[
            [ 5.,  6.,  7.,  8.],
            [13., 14., 15., 16.]
        ].concat(),
        &[1, 2, 4, 1],
    );

    #[rustfmt::skip]    
    test_slice(
        &[
            [ 1.,  2.,  3.,  4.], 
            [ 5.,  6.,  7.,  8.],
            [ 9., 10., 11., 12.], 
            [13., 14., 15., 16.]
        ].concat(),
        &[1, 4, 4, 1],
        &[0],
        &[2],
        Some(&[1]),
        Some(&[2]),
        &[
            [ 1.,  2.,  3.,  4.], 
        ].concat(),
        &[1, 1, 4, 1],
    );

    #[rustfmt::skip]    
    test_slice(
        &[
            [ 1.,  2.,  3.,  4.], 
            [ 5.,  6.,  7.,  8.],
            [ 9., 10., 11., 12.], 
            [13., 14., 15., 16.]
        ].concat(),
        &[1, 4, 4, 1],
        &[3],
        &[4],
        Some(&[1]),
        Some(&[2]),
        &[
            [13., 14., 15., 16.]
        ].concat(),
        &[1, 1, 4, 1],
    );


    #[rustfmt::skip]    
    test_slice(
        &[
            [ 1.,  2.,  3.,  4.], 
            [ 5.,  6.,  7.,  8.],
            [ 9., 10., 11., 12.], 
            [13., 14., 15., 16.]
        ].concat(),
        &[1, 4, 4, 1],
        &[0],
        &[4],
        Some(&[2]),
        Some(&[2]),
        &[
            [ 1.,  3.], 
            [ 5.,  7.],
            [ 9., 11.], 
            [13., 15.]
        ].concat(),
        &[1, 4, 2, 1],
    );

    #[rustfmt::skip]    
    test_slice(
        &[
            [ 1.,  2.,  3.,  4.], 
            [ 5.,  6.,  7.,  8.],
            [ 9., 10., 11., 12.], 
            [13., 14., 15., 16.]
        ].concat(),
        &[1, 4, 4, 1],
        &[0],
        &[2],
        Some(&[2]),
        Some(&[2]),
        &[
            [ 1.], 
            [ 5.],
            [ 9.], 
            [13.]
        ].concat(),
        &[1, 4, 1, 1],
    );



    #[rustfmt::skip]    
    test_slice(
        &[
            [ 1.,  2.,  3.,  4.], 
            [ 5.,  6.,  7.,  8.],
            [ 9., 10., 11., 12.], 
            [13., 14., 15., 16.]
        ].concat(),
        &[1, 4, 4, 1],
        &[2],
        &[4],
        Some(&[2]),
        Some(&[2]),
        &[
            [ 3.], 
            [ 7.],
            [11.], 
            [15.]
        ].concat(),
        &[1, 4, 1, 1],
    );
}


#[test]
fn slice_1x3x640x640_step2() {
    let input = vec![1.0; 1 * 3 * 640 * 640];
    let output = vec![1.0; 1 * 3 * 320 * 640];
    #[rustfmt::skip]    
    test_slice(
        &input,
        &[1, 3, 640, 640],
        &[0],
        &[640],
        Some(&[2]),
        Some(&[2]),
        &output,
        &[1, 3, 320, 640],
    );
}


#[test]
fn slice_none_axes_and_steps() {
    #[rustfmt::skip]    
    test_slice(
        &[
            [ 1.,  2.,  3.,  4.], 
            [ 5.,  6.,  7.,  8.],
            [ 9., 10., 11., 12.], 
            [13., 14., 15., 16.]
        ].concat(),
        &[4, 4, 1],
        &[0],
        &[2],
        None,
        None,
        &[
            [ 1.,  2.,  3.,  4.], 
            [ 5.,  6.,  7.,  8.],
        ].concat(),
        &[2, 4, 1],
    );
}




#[test]
fn slice_ends_max_i32() {
    #[rustfmt::skip]    
    test_slice(
        &[
            [ 1.,  2.,  3.,  4.], 
            [ 5.,  6.,  7.,  8.],
            [ 9., 10., 11., 12.], 
            [13., 14., 15., 16.]
        ].concat(),
        &[4, 4, 1],
        &[0],
        &[i32::MAX],
        Some(&[0]),
        Some(&[2]),
        &[
            [ 1.,  2.,  3.,  4.], 
            [ 9., 10., 11., 12.], 
        ].concat(),
        &[2, 4, 1],
    );


    #[rustfmt::skip]    
    test_slice(
        &[
            [ 1.,  2.,  3.,  4.], 
            [ 5.,  6.,  7.,  8.],
            [ 9., 10., 11., 12.], 
            [13., 14., 15., 16.]
        ].concat(),
        &[4, 4, 1],
        &[0],
        &[i32::MAX],
        Some(&[1]),
        Some(&[2]),
        &[
            [ 1.,  3.], 
            [ 5.,  7.],
            [ 9., 11.], 
            [13., 15.]
        ].concat(),
        &[4, 2, 1],
    );
}
