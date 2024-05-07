use std::{collections::HashMap, convert::TryInto, vec};
use wonnx::{
    onnx::AttributeProto,
    utils::{attribute, graph, model, node, tensor},
};
mod common;

#[test]
fn test_concat() {
    let n: usize = 16;

    let xdata: Vec<f32> = (0..n).map(|x| x as f32).collect();
    let mut ydata: Vec<f32> = (n..2 * n).map(|x| x as f32).collect();
    let input_dims = vec![n as i64];
    let output_dims = vec![(n * 2) as i64];

    let input_data = HashMap::from([
        ("X".into(), xdata.as_slice().into()),
        ("Y".into(), ydata.as_slice().into()),
    ]);

    let model = model(graph(
        vec![tensor("X", &input_dims), tensor("Y", &input_dims)],
        vec![tensor("Z", &output_dims)],
        vec![],
        vec![],
        vec![node(vec!["X", "Y"], vec!["Z"], "a", "Concat", vec![])],
    ));

    let session =
        pollster::block_on(wonnx::Session::from_model(model)).expect("Session did not create");

    let result = pollster::block_on(session.run(&input_data)).unwrap();

    let mut expected_result = xdata.clone();
    expected_result.append(&mut ydata);

    common::assert_eq_vector((&result["Z"]).try_into().unwrap(), &expected_result);
}

#[test]
fn test_concat_long() {
    let n: usize = 100000;

    let xdata: Vec<f32> = (0..n).map(|x| x as f32).collect();
    let mut ydata: Vec<f32> = (n..2 * n).map(|x| x as f32).collect();
    let input_dims = vec![n as i64];
    let output_dims = vec![(n * 2) as i64];

    let input_data = HashMap::from([
        ("X".into(), xdata.as_slice().into()),
        ("Y".into(), ydata.as_slice().into()),
    ]);

    let model = model(graph(
        vec![tensor("X", &input_dims), tensor("Y", &input_dims)],
        vec![tensor("Z", &output_dims)],
        vec![],
        vec![],
        vec![node(vec!["X", "Y"], vec!["Z"], "a", "Concat", vec![])],
    ));

    let session =
        pollster::block_on(wonnx::Session::from_model(model)).expect("Session did not create");

    let result = pollster::block_on(session.run(&input_data)).unwrap();

    let mut expected_result = xdata.clone();
    expected_result.append(&mut ydata);

    common::assert_eq_vector((&result["Z"]).try_into().unwrap(), &expected_result);
}

#[test]
fn test_concat4() {
    let n: usize = 13;

    let xdata: Vec<f32> = (0..n).map(|x| x as f32).collect();
    let mut ydata: Vec<f32> = (n..2 * n).map(|x| x as f32).collect();
    let mut zdata: Vec<f32> = (n * 2..3 * n).map(|x| x as f32).collect();
    let mut wdata: Vec<f32> = (n * 3..4 * n).map(|x| x as f32).collect();
    let input_dims = vec![n as i64];
    let output_dims = vec![(n * 4) as i64];

    let input_data = HashMap::from([
        ("X".into(), xdata.as_slice().into()),
        ("Y".into(), ydata.as_slice().into()),
        ("Z".into(), zdata.as_slice().into()),
        ("W".into(), wdata.as_slice().into()),
    ]);

    let model = model(graph(
        vec![
            tensor("X", &input_dims),
            tensor("Y", &input_dims),
            tensor("Z", &input_dims),
            tensor("W", &input_dims),
        ],
        vec![tensor("O", &output_dims)],
        vec![],
        vec![],
        vec![node(
            vec!["X", "Y", "Z", "W"],
            vec!["O"],
            "a",
            "Concat",
            vec![],
        )],
    ));

    let session =
        pollster::block_on(wonnx::Session::from_model(model)).expect("Session did not create");

    let result = pollster::block_on(session.run(&input_data)).unwrap();

    let mut expected_result = xdata.clone();
    expected_result.append(&mut ydata);
    expected_result.append(&mut zdata);
    expected_result.append(&mut wdata);

    common::assert_eq_vector((&result["O"]).try_into().unwrap(), &expected_result);
}

#[test]
fn test_concat_axis1() {
    let xdata = vec![
        vec![vec![1., 2.], vec![3., 4.]].concat(),
        vec![vec![5., 6.], vec![7., 8.]].concat(),
    ]
    .concat();
    let x_input_dims = vec![2, 2, 2];
    let ydata = vec![
        vec![vec![9., 10.], vec![11., 12.]].concat(),
        vec![vec![13., 14.], vec![15., 16.]].concat(),
    ]
    .concat();
    let y_input_dims = vec![2, 2, 2];
    let zdata = vec![
        vec![vec![17., 18.], vec![19., 20.]].concat(),
        vec![vec![21., 22.], vec![23., 24.]].concat(),
    ]
    .concat();
    let z_input_dims = vec![2, 2, 2];

    let input_data = HashMap::from([
        ("X".into(), xdata.as_slice().into()),
        ("Y".into(), ydata.as_slice().into()),
        ("Z".into(), zdata.as_slice().into()),
    ]);

    let attributes: Vec<AttributeProto> = vec![attribute("axis", 1)];

    let model = model(graph(
        vec![
            tensor("X", &x_input_dims),
            tensor("Y", &y_input_dims),
            tensor("Z", &z_input_dims),
        ],
        vec![tensor("W", &vec![2, 6, 2])],
        vec![],
        vec![],
        vec![node(
            vec!["X", "Y", "Z"],
            vec!["W"],
            "a",
            "Concat",
            attributes,
        )],
    ));

    let session =
        pollster::block_on(wonnx::Session::from_model(model)).expect("Session did not create");
    let result = pollster::block_on(session.run(&input_data)).unwrap();

    let expected_result = vec![
        vec![
            vec![1., 2.],
            vec![3., 4.],
            vec![9., 10.],
            vec![11., 12.],
            vec![17., 18.],
            vec![19., 20.],
        ]
        .concat(),
        vec![
            vec![5., 6.],
            vec![7., 8.],
            vec![13., 14.],
            vec![15., 16.],
            vec![21., 22.],
            vec![23., 24.],
        ]
        .concat(),
    ]
    .concat();

    common::assert_eq_vector((&result["W"]).try_into().unwrap(), &expected_result);
}

#[test]
fn test_concat_axis2() {
    let xdata = vec![
        vec![
            vec![1., 2., 3., 4.],
            vec![5., 6., 7., 8.],
            vec![9., 10., 11., 12.],
        ]
        .concat(),
        vec![
            vec![13., 14., 15., 16.],
            vec![17., 18., 19., 20.],
            vec![21., 22., 23., 24.],
        ]
        .concat(),
    ]
    .concat();
    let x_input_dims = vec![1, 2, 3, 4];
    let ydata = vec![
        vec![
            vec![25., 26., 27., 28.],
            vec![29., 30., 31., 32.],
            vec![33., 34., 35., 36.],
        ]
        .concat(),
        vec![
            vec![37., 38., 39., 40.],
            vec![41., 42., 43., 44.],
            vec![45., 46., 47., 48.],
        ]
        .concat(),
    ]
    .concat();
    let y_input_dims = vec![1, 2, 3, 4];
    let zdata = vec![
        vec![
            vec![49., 50., 51., 52.],
            vec![53., 54., 55., 56.],
            vec![57., 58., 59., 60.],
        ]
        .concat(),
        vec![
            vec![61., 62., 63., 64.],
            vec![65., 66., 67., 68.],
            vec![69., 70., 71., 72.],
        ]
        .concat(),
    ]
    .concat();
    let z_input_dims = vec![1, 2, 3, 4];

    let input_data = HashMap::from([
        ("X".into(), xdata.as_slice().into()),
        ("Y".into(), ydata.as_slice().into()),
        ("Z".into(), zdata.as_slice().into()),
    ]);

    let attributes: Vec<AttributeProto> = vec![attribute("axis", 2)];

    let model = model(graph(
        vec![
            tensor("X", &x_input_dims),
            tensor("Y", &y_input_dims),
            tensor("Z", &z_input_dims),
        ],
        vec![tensor("W", &vec![1, 2, 9, 4])],
        vec![],
        vec![],
        vec![node(
            vec!["X", "Y", "Z"],
            vec!["W"],
            "a",
            "Concat",
            attributes,
        )],
    ));

    let session =
        pollster::block_on(wonnx::Session::from_model(model)).expect("Session did not create");
    let result = pollster::block_on(session.run(&input_data)).unwrap();

    // 2x6x3x4
    let expected_result = vec![
        vec![
            vec![1., 2., 3., 4.],
            vec![5., 6., 7., 8.],
            vec![9., 10., 11., 12.],
            vec![25., 26., 27., 28.],
            vec![29., 30., 31., 32.],
            vec![33., 34., 35., 36.],
            vec![49., 50., 51., 52.],
            vec![53., 54., 55., 56.],
            vec![57., 58., 59., 60.],
        ]
        .concat(),
        vec![
            vec![13., 14., 15., 16.],
            vec![17., 18., 19., 20.],
            vec![21., 22., 23., 24.],
            vec![37., 38., 39., 40.],
            vec![41., 42., 43., 44.],
            vec![45., 46., 47., 48.],
            vec![61., 62., 63., 64.],
            vec![65., 66., 67., 68.],
            vec![69., 70., 71., 72.],
        ]
        .concat(),
    ]
    .concat();
    common::assert_eq_vector((&result["W"]).try_into().unwrap(), &expected_result);
}

#[test]
fn test_concat_axis3() {
    let xdata = vec![
        vec![
            vec![1., 2., 3., 4.],
            vec![5., 6., 7., 8.],
            vec![9., 10., 11., 12.],
        ]
        .concat(),
        vec![
            vec![13., 14., 15., 16.],
            vec![17., 18., 19., 20.],
            vec![21., 22., 23., 24.],
        ]
        .concat(),
    ]
    .concat();
    let x_input_dims = vec![1, 2, 3, 4];
    let ydata = vec![
        vec![
            vec![25., 26., 27., 28.],
            vec![29., 30., 31., 32.],
            vec![33., 34., 35., 36.],
        ]
        .concat(),
        vec![
            vec![37., 38., 39., 40.],
            vec![41., 42., 43., 44.],
            vec![45., 46., 47., 48.],
        ]
        .concat(),
    ]
    .concat();
    let y_input_dims = vec![1, 2, 3, 4];
    let zdata = vec![
        vec![
            vec![49., 50., 51., 52.],
            vec![53., 54., 55., 56.],
            vec![57., 58., 59., 60.],
        ]
        .concat(),
        vec![
            vec![61., 62., 63., 64.],
            vec![65., 66., 67., 68.],
            vec![69., 70., 71., 72.],
        ]
        .concat(),
    ]
    .concat();
    let z_input_dims = vec![1, 2, 3, 4];

    let input_data = HashMap::from([
        ("X".into(), xdata.as_slice().into()),
        ("Y".into(), ydata.as_slice().into()),
        ("Z".into(), zdata.as_slice().into()),
    ]);

    let attributes: Vec<AttributeProto> = vec![attribute("axis", 3)];

    let model = model(graph(
        vec![
            tensor("X", &x_input_dims),
            tensor("Y", &y_input_dims),
            tensor("Z", &z_input_dims),
        ],
        vec![tensor("W", &vec![1, 2, 3, 12])],
        vec![],
        vec![],
        vec![node(
            vec!["X", "Y", "Z"],
            vec!["W"],
            "a",
            "Concat",
            attributes,
        )],
    ));

    let session =
        pollster::block_on(wonnx::Session::from_model(model)).expect("Session did not create");
    let result = pollster::block_on(session.run(&input_data)).unwrap();

    // 2x6x3x4
    let expected_result = vec![
        vec![
            vec![1., 2., 3., 4., 25., 26., 27., 28., 49., 50., 51., 52.],
            vec![5., 6., 7., 8., 29., 30., 31., 32., 53., 54., 55., 56.],
            vec![9., 10., 11., 12., 33., 34., 35., 36., 57., 58., 59., 60.],
        ]
        .concat(),
        vec![
            vec![13., 14., 15., 16., 37., 38., 39., 40., 61., 62., 63., 64.],
            vec![17., 18., 19., 20., 41., 42., 43., 44., 65., 66., 67., 68.],
            vec![21., 22., 23., 24., 45., 46., 47., 48., 69., 70., 71., 72.],
        ]
        .concat(),
    ]
    .concat();
    common::assert_eq_vector((&result["W"]).try_into().unwrap(), &expected_result);
}
