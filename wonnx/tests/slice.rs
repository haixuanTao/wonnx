use std::{collections::HashMap, convert::TryInto, vec};
use wonnx::{
    onnx::AttributeProto,
    utils::{ attribute, graph, model, node, tensor, InputTensor},
};
mod common;

fn test_slice(
    input: &[f32],
    input_shape: &[i64],
    starts: &Vec<i64>,
    ends: &Vec<i64>,
    axes: &Vec<i64>,
    steps: &Vec<i64>,
    output: &[f32],
    output_shape: &[i64],
) {
    let mut input_data = HashMap::<String, InputTensor>::new();
    input_data.insert("X".to_string(), input.into());

    let attributes: Vec<AttributeProto> = vec![
        attribute("starts", starts.clone()),
        attribute("ends", ends.clone()),
        attribute("axes", axes.clone()),
        attribute("steps", steps.clone()),
    ];

    let model = model(graph(
        vec![tensor("X", input_shape)],
        vec![tensor("Y", output_shape)],
        vec![],
        vec![],
        vec![node(
            vec!["X"],
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
        &vec![0],
        &vec![1],
        &vec![0],
        &vec![1],
        &[[1., 2., 3., 4.], [5., 6., 7., 8.]].concat(),
        &[1, 2, 4],
    );

    // axes == 1
    test_slice(
        &[[1., 2., 3., 4.], [5., 6., 7., 8.]].concat(),
        &[1, 2, 4],
        &vec![0],
        &vec![2],
        &vec![1],
        &vec![1],
        &[[1., 2., 3., 4.], [5., 6., 7., 8.]].concat(),
        &[1, 2, 4],
    );

    // axes == 2
    test_slice(
        &[[1., 2., 3., 4.], [5., 6., 7., 8.]].concat(),
        &[1, 2, 4],
        &vec![0],
        &vec![4],
        &vec![2],
        &vec![1],
        &[[1., 2., 3., 4.], [5., 6., 7., 8.]].concat(),
        &[1, 2, 4],
    );

    // axes == 1
    test_slice(
        &[[1., 2., 3., 4.], [5., 6., 7., 8.]].concat(),
        &[1, 2, 4],
        &vec![0],
        &vec![1],
        &vec![1],
        &vec![1],
        &[[1., 2., 3., 4.]].concat(),
        &[1, 1, 4],
    );

    // axes == 1
    test_slice(
        &[[1., 2., 3., 4.], [5., 6., 7., 8.]].concat(),
        &[1, 2, 4],
        &vec![1],
        &vec![2],
        &vec![1],
        &vec![1],
        &[[5., 6., 7., 8.]].concat(),
        &[1, 1, 4],
    );

    // axes == 2
    test_slice(
        &[[1., 2., 3., 4.], [5., 6., 7., 8.]].concat(),
        &[1, 2, 4],
        &vec![0],
        &vec![2],
        &vec![2],
        &vec![1],
        &[[1., 2.], [5., 6.]].concat(),
        &[1, 2, 2],
    );

    // axes == 2
    test_slice(
        &[[1., 2., 3., 4.], [5., 6., 7., 8.]].concat(),
        &[1, 2, 4],
        &vec![2],
        &vec![4],
        &vec![2],
        &vec![1],
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
        &vec![0],
        &vec![1],
        &vec![0],
        &vec![2],
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
        &vec![0],
        &vec![4],
        &vec![1],
        &vec![2],
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
        &vec![1],
        &vec![4],
        &vec![1],
        &vec![2],
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
        &vec![0],
        &vec![2],
        &vec![1],
        &vec![2],
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
        &vec![3],
        &vec![4],
        &vec![1],
        &vec![2],
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
        &vec![0],
        &vec![4],
        &vec![2],
        &vec![2],
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
        &vec![0],
        &vec![2],
        &vec![2],
        &vec![2],
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
        &vec![2],
        &vec![4],
        &vec![2],
        &vec![2],
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
fn slice_1x3x416x416_step2() {
    let input0 = vec![1.0; 1 * 3 * 416 * 416];
    let output0 = vec![1.0; 1 * 3 * 208 * 416];
    #[rustfmt::skip]    
    test_slice(
        &input0,
        &[1, 3, 416, 416],
        &vec![0],
        &vec![i32::MAX as i64],
        &vec![2],
        &vec![2],
        &output0,
        &[1, 3, 208, 416],
    );
}

#[test]
fn slice_1x3x416x416_step2_axes2_start0() {
    let mut input2 = vec![];
    for _ in 0..3 {
        let mut row = vec![];
        for j in 0..416 {
            let mut col = vec![];
            for _ in 0..416 {
                if j % 2 == 0 {
                    col.push(1.0);
                } else {
                    col.push(0.0);
                
                }
            }
            row.push(col);
        }
        input2.push(row);
    }
    // flatten the input
    let input2 = input2.iter().flatten().flatten().copied().collect::<Vec<f32>>();


    let output2 = vec![1.0; 1 * 3 * 208 * 416];
    #[rustfmt::skip]    
    test_slice(
        &input2,
        &[1, 3, 416, 416],
        &vec![0],
        &vec![i32::MAX as i64],
        &vec![2],
        &vec![2],
        &output2,
        &[1, 3, 208, 416],
    );
}


#[test]
fn slice_1x3x416x416_step2_axes2_start1() {
    let mut input2 = vec![];
    for _ in 0..3 {
        let mut row = vec![];
        for j in 0..416 {
            let mut col = vec![];
            for _ in 0..416 {
                if j % 2 == 0 {
                    col.push(0.0);
                } else {
                    col.push(1.0);
                
                }
            }
            row.push(col);
        }
        input2.push(row);
    }
    // flatten the input
    let input2 = input2.iter().flatten().flatten().copied().collect::<Vec<f32>>();


    let output2 = vec![1.0; 1 * 3 * 208 * 416];
    #[rustfmt::skip]    
    test_slice(
        &input2,
        &[1, 3, 416, 416],
        &vec![1],
        &vec![i32::MAX as i64],
        &vec![2],
        &vec![2],
        &output2,
        &[1, 3, 208, 416],
    );
}

#[test]
fn slice_1x3x8x8_step2_axes2_start1() {
    let input2 = vec![
        vec![
            vec![1., 1., 1., 1., 1., 1., 1., 1.],
            vec![0., 0., 0., 0., 0., 0., 0., 0.],
            vec![1., 1., 1., 1., 1., 1., 1., 1.],
            vec![0., 0., 0., 0., 0., 0., 0., 0.],
            vec![1., 1., 1., 1., 1., 1., 1., 1.],
            vec![0., 0., 0., 0., 0., 0., 0., 0.],
            vec![1., 1., 1., 1., 1., 1., 1., 1.],
            vec![0., 0., 0., 0., 0., 0., 0., 0.],
        ],
        vec![
            vec![1., 1., 1., 1., 1., 1., 1., 1.],
            vec![0., 0., 0., 0., 0., 0., 0., 0.],
            vec![1., 1., 1., 1., 1., 1., 1., 1.],
            vec![0., 0., 0., 0., 0., 0., 0., 0.],
            vec![1., 1., 1., 1., 1., 1., 1., 1.],
            vec![0., 0., 0., 0., 0., 0., 0., 0.],
            vec![1., 1., 1., 1., 1., 1., 1., 1.],
            vec![0., 0., 0., 0., 0., 0., 0., 0.],
        ],
        vec![
            vec![1., 1., 1., 1., 1., 1., 1., 1.],
            vec![0., 0., 0., 0., 0., 0., 0., 0.],
            vec![1., 1., 1., 1., 1., 1., 1., 1.],
            vec![0., 0., 0., 0., 0., 0., 0., 0.],
            vec![1., 1., 1., 1., 1., 1., 1., 1.],
            vec![0., 0., 0., 0., 0., 0., 0., 0.],
            vec![1., 1., 1., 1., 1., 1., 1., 1.],
            vec![0., 0., 0., 0., 0., 0., 0., 0.],
        ]
    ];

    // flatten the input
    let input2 = input2.iter().flatten().flatten().copied().collect::<Vec<f32>>();

    let output2 = vec![0.0; 1 * 3 * 4 * 8];
    #[rustfmt::skip]
    test_slice(
        &input2,
        &[1, 3, 8, 8],
        &vec![1],
        &vec![i32::MAX as i64],
        &vec![2],
        &vec![2],
        &output2,
        &[1, 3, 4, 8],
    );
  

}

#[test]
fn slice_1x3x8x8_step2_axes3_start1() {
    let input2 = vec![
        vec![
            vec![0., 1., 0., 1., 0., 1., 0., 1.],
            vec![0., 1., 0., 1., 0., 1., 0., 1.],
            vec![0., 1., 0., 1., 0., 1., 0., 1.],
            vec![0., 1., 0., 1., 0., 1., 0., 1.],
            vec![0., 1., 0., 1., 0., 1., 0., 1.],
            vec![0., 1., 0., 1., 0., 1., 0., 1.],
            vec![0., 1., 0., 1., 0., 1., 0., 1.],
            vec![0., 1., 0., 1., 0., 1., 0., 1.],
        ],
        vec![
            vec![0., 1., 0., 1., 0., 1., 0., 1.],
            vec![0., 1., 0., 1., 0., 1., 0., 1.],
            vec![0., 1., 0., 1., 0., 1., 0., 1.],
            vec![0., 1., 0., 1., 0., 1., 0., 1.],
            vec![0., 1., 0., 1., 0., 1., 0., 1.],
            vec![0., 1., 0., 1., 0., 1., 0., 1.],
            vec![0., 1., 0., 1., 0., 1., 0., 1.],
            vec![0., 1., 0., 1., 0., 1., 0., 1.],
        ],
        vec![
            vec![0., 1., 0., 1., 0., 1., 0., 1.],
            vec![0., 1., 0., 1., 0., 1., 0., 1.],
            vec![0., 1., 0., 1., 0., 1., 0., 1.],
            vec![0., 1., 0., 1., 0., 1., 0., 1.],
            vec![0., 1., 0., 1., 0., 1., 0., 1.],
            vec![0., 1., 0., 1., 0., 1., 0., 1.],
            vec![0., 1., 0., 1., 0., 1., 0., 1.],
            vec![0., 1., 0., 1., 0., 1., 0., 1.],
        ]
    ];
    // flatten the input
    let input2 = input2.iter().flatten().flatten().copied().collect::<Vec<f32>>();

    let output2 = vec![1.0; 1 * 3 * 8 * 4];
    #[rustfmt::skip]
    test_slice(
        &input2,
        &[1, 3, 8, 8],
        &vec![1],
        &vec![i32::MAX as i64],
        &vec![3],
        &vec![2],
        &output2,
        &[1, 3, 8, 4],
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
        &vec![0],
        &vec![2],
        &vec![0],
        &vec![1],
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
        &vec![0],
        &vec![i32::MAX as i64],
        &vec![0],
        &vec![2],
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
        &vec![0],
        &vec![i32::MAX as i64],
        &vec![1],
        &vec![2],
        &[
            [ 1.,  3.], 
            [ 5.,  7.],
            [ 9., 11.], 
            [13., 15.]
        ].concat(),
        &[4, 2, 1],
    );
}
