//! Conversion of ONNX models to WONNX IR
use crate::gpu::GpuModel;
use crate::ir::AttributeValue;
use crate::ir::Input;
use crate::ir::IrError;
use crate::ir::Node;
use crate::ir::NodeDefinition;
use crate::ir::OperatorDefinition;
use crate::ir::Tensor;
use crate::onnx;
use crate::onnx::AttributeProto;
use crate::onnx::AttributeProto_AttributeType;
use crate::onnx::GraphProto;
use crate::onnx::ModelProto;
use crate::onnx::NodeProto;
use crate::onnx::OperatorSetIdProto;
use crate::onnx::TensorProto;
use crate::onnx::TensorProto_DataType;
use crate::onnx::TensorShapeProto;
use crate::onnx::TensorShapeProto_Dimension;
use crate::onnx::TypeProto;
use crate::onnx::TypeProto_Tensor;
use crate::onnx::TypeProto_oneof_value;
use crate::onnx::ValueInfoProto;
use crate::optimizer::Optimizer;
use crate::resource::request_device_queue;
use crate::tensor::DataTypeError;
use crate::tensor::ScalarType;
use crate::tensor::Shape;
use crate::tensor::TensorData;
use crate::Session;
use crate::SessionConfig;
use crate::SessionError;
use protobuf::Message;
use protobuf::ProtobufEnum;
use protobuf::RepeatedField;
use std::borrow::Cow;
use std::collections::HashMap;
use std::convert::From;
use std::convert::Into;
use std::convert::TryFrom;
use std::path::Path;
use std::str::from_utf8;
use std::sync::Arc;
use thiserror::Error;

impl TensorProto {
    pub fn from(value: TensorData, dims: Vec<i64>) -> Self {
        let mut tensor = TensorProto::new();
        match value {
            TensorData::F32(v) => {
                tensor.set_data_type(ScalarType::F32.to_onnx_datatype().value());
                tensor.set_float_data(v.to_vec());
            }
            TensorData::I32(v) => {
                tensor.set_data_type(ScalarType::I32.to_onnx_datatype().value());
                tensor.set_int32_data(v.to_vec());
            }
            TensorData::I64(v) => {
                tensor.set_data_type(ScalarType::I64.to_onnx_datatype().value());
                tensor.set_int64_data(v.to_vec());
            }
            TensorData::U8(v) => {
                tensor.set_data_type(ScalarType::U8.to_onnx_datatype().value());
                tensor.set_raw_data(v.to_vec());
            }
        }
        tensor.set_dims(dims);
        tensor
    }
}

impl<'a> TryFrom<&'a TensorProto> for TensorData<'a> {
    type Error = DataTypeError;

    fn try_from(value: &'a TensorProto) -> Result<Self, Self::Error> {
        Ok(match ScalarType::from_onnx_i32(value.get_data_type())? {
            ScalarType::F32 => TensorData::F32(Cow::Borrowed(value.get_float_data())),
            ScalarType::I64 => TensorData::I64(Cow::Borrowed(value.get_int64_data())),
            ScalarType::I32 => TensorData::I32(Cow::Borrowed(value.get_int32_data())),
            ScalarType::U8 => TensorData::U8(Cow::Borrowed(value.get_raw_data())),
        })
    }
}

impl ScalarType {
    pub fn from_onnx_i32(onnx: i32) -> Result<ScalarType, DataTypeError> {
        let onnx_dt =
            TensorProto_DataType::from_i32(onnx).ok_or(DataTypeError::NotRecognized(onnx))?;
        Self::from(onnx_dt)
    }

    pub fn from(onnx: TensorProto_DataType) -> Result<ScalarType, DataTypeError> {
        Ok(match onnx {
            TensorProto_DataType::FLOAT => ScalarType::F32,
            TensorProto_DataType::INT64 => ScalarType::I64,
            TensorProto_DataType::INT32 => ScalarType::I32,
            TensorProto_DataType::UINT8 => ScalarType::U8,
            _ => return Err(DataTypeError::NotSupported(onnx.value())),
        })
    }

    pub fn to_onnx_datatype(&self) -> TensorProto_DataType {
        match self {
            ScalarType::F32 => TensorProto_DataType::FLOAT,
            ScalarType::I64 => TensorProto_DataType::INT64,
            ScalarType::I32 => TensorProto_DataType::INT32,
            ScalarType::U8 => TensorProto_DataType::UINT8,
        }
    }
}

impl ValueInfoProto {
    pub fn get_shape(&self) -> Result<Shape, DataTypeError> {
        Ok(match &self.get_field_type().value {
            Some(t) => match t {
                onnx::TypeProto_oneof_value::tensor_type(tensor_proto) => Shape::from(
                    ScalarType::from_onnx_i32(tensor_proto.get_elem_type())?,
                    self.get_field_type()
                        .get_tensor_type()
                        .get_shape()
                        .get_dim()
                        .iter()
                        .map(|x| {
                            if x.has_dim_param() {
                                return Err(DataTypeError::ParametrizedDimensionUnsupported(
                                    x.get_dim_param().to_string(),
                                ));
                            }
                            Ok(x.get_dim_value() as usize)
                        })
                        .collect::<Result<Vec<usize>, DataTypeError>>()?
                        .as_slice(),
                ),
                onnx::TypeProto_oneof_value::sequence_type(_) => todo!(),
                onnx::TypeProto_oneof_value::map_type(_) => todo!(),
                onnx::TypeProto_oneof_value::optional_type(_) => todo!(),
                onnx::TypeProto_oneof_value::sparse_tensor_type(_) => todo!(),
            },
            None => return Err(DataTypeError::Undefined),
        })
    }

    pub fn set_shape(&mut self, shape: &Shape) {
        let mut tpt = TypeProto_Tensor::new();
        tpt.set_elem_type(shape.data_type.to_onnx_datatype().value());

        let mut tsp = TensorShapeProto::new();
        tsp.dim.extend(shape.dims.iter().map(|x| {
            let mut tspd = TensorShapeProto_Dimension::new();
            tspd.set_dim_value(*x as i64);
            tspd
        }));
        tpt.set_shape(tsp);

        let mut tp = TypeProto::new();
        tp.value = Some(TypeProto_oneof_value::tensor_type(tpt));
        self.set_field_type(tp);
    }
}

/// Shorthand method to define an ONNX tensor with the specified name and shape (data type is f32)
pub fn onnx_tensor(name: &str, dimensions: &[i64]) -> onnx::ValueInfoProto {
    onnx_tensor_of_type(name, dimensions, TensorProto_DataType::FLOAT)
}

/// Shorthand method to define an ONNX tensor with the specified name, shape and data type
pub fn onnx_tensor_of_type(
    name: &str,
    dimensions: &[i64],
    data_type: TensorProto_DataType,
) -> onnx::ValueInfoProto {
    let mut dim_value = vec![];
    for dimension in dimensions {
        let mut dim_channel = onnx::TensorShapeProto_Dimension::new();
        dim_channel.set_dim_value(*dimension);
        dim_value.push(dim_channel);
    }

    let mut shape_tensor_proto = onnx::TensorShapeProto::new();
    shape_tensor_proto.set_dim(protobuf::RepeatedField::from(dim_value));

    let mut type_proto_tensor = onnx::TypeProto_Tensor::new();
    type_proto_tensor.set_elem_type(data_type.value());
    type_proto_tensor.set_shape(shape_tensor_proto);

    let mut type_proto = onnx::TypeProto::new();
    type_proto.set_tensor_type(type_proto_tensor);

    let mut tensor = onnx::ValueInfoProto::new();
    tensor.set_name(name.to_string());
    tensor.set_field_type(type_proto);

    tensor
}

pub fn onnx_initializer(name: &str, data: Vec<f32>, dimensions: Vec<i64>) -> onnx::TensorProto {
    let mut initializer = crate::onnx::TensorProto::new();
    assert_eq!(
        dimensions.iter().cloned().product::<i64>() as usize,
        data.len()
    );
    initializer.set_dims(dimensions);
    initializer.set_name(name.to_string());
    initializer.set_data_type(TensorProto_DataType::FLOAT.value());
    initializer.set_float_data(data);
    initializer
}

pub fn onnx_initializer_int64(
    name: &str,
    data: Vec<i64>,
    dimensions: Vec<i64>,
) -> onnx::TensorProto {
    let mut initializer = crate::onnx::TensorProto::new();
    assert_eq!(
        dimensions.iter().cloned().product::<i64>() as usize,
        data.len()
    );
    initializer.set_name(name.to_string());
    initializer.set_dims(dimensions);
    initializer.set_data_type(TensorProto_DataType::INT64.value());
    initializer.set_int64_data(data);
    initializer
}

pub fn onnx_attribute(name: &str, inputs: impl Into<onnx::AttributeProto>) -> onnx::AttributeProto {
    let mut attributes: onnx::AttributeProto = inputs.into();
    attributes.set_name(name.to_string());
    attributes
}

/// Create a node - the node name will be set to the name of the first output
pub fn onnx_node(
    inputs: Vec<&str>,
    outputs: Vec<&str>,
    op_type: &str,
    attributes: Vec<onnx::AttributeProto>,
) -> onnx::NodeProto {
    let mut node = crate::onnx::NodeProto::new();

    node.set_op_type(op_type.to_string());
    node.set_name(outputs[0].to_string());
    node.set_input(protobuf::RepeatedField::from(
        inputs
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<String>>(),
    ));
    node.set_output(protobuf::RepeatedField::from(
        outputs
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<String>>(),
    ));
    node.set_attribute(protobuf::RepeatedField::from(attributes));
    node
}

pub fn onnx_graph(
    inputs: Vec<onnx::ValueInfoProto>,
    outputs: Vec<onnx::ValueInfoProto>,
    mut infos: Vec<onnx::ValueInfoProto>,
    initializers: Vec<onnx::TensorProto>,
    nodes: Vec<onnx::NodeProto>,
) -> onnx::GraphProto {
    let mut graph = onnx::GraphProto::new();
    graph.set_node(protobuf::RepeatedField::from(nodes));
    graph.set_input(protobuf::RepeatedField::from(inputs));
    graph.set_output(protobuf::RepeatedField::from(outputs));

    // Auto-generate tensor information for initializers so users don't have to specify those
    for i in &initializers {
        infos.push(onnx_tensor_of_type(
            i.get_name(),
            i.get_dims(),
            onnx::TensorProto_DataType::from_i32(i.get_data_type()).unwrap(),
        ));
    }

    graph.set_initializer(protobuf::RepeatedField::from(initializers));
    graph.set_value_info(protobuf::RepeatedField::from(infos));
    graph
}

pub fn onnx_model_with_opset(graph: onnx::GraphProto, opset_version: i64) -> onnx::ModelProto {
    let mut model = crate::onnx::ModelProto::new();
    let mut onnx_opset_import = OperatorSetIdProto::new();
    onnx_opset_import.set_domain("".to_string());
    onnx_opset_import.set_version(opset_version);
    model.set_opset_import(RepeatedField::from_slice(&[onnx_opset_import]));
    model.set_graph(graph);
    model
}

pub fn onnx_model(graph: onnx::GraphProto) -> onnx::ModelProto {
    onnx_model_with_opset(graph, 13)
}

impl From<Vec<i64>> for onnx::AttributeProto {
    fn from(value: Vec<i64>) -> Self {
        let mut attributes = crate::onnx::AttributeProto::new();
        attributes.set_ints(value);
        attributes.set_field_type(AttributeProto_AttributeType::INTS);
        attributes
    }
}

impl From<Vec<f32>> for onnx::AttributeProto {
    fn from(value: Vec<f32>) -> Self {
        let mut attributes = crate::onnx::AttributeProto::new();
        attributes.set_floats(value);
        attributes.set_field_type(AttributeProto_AttributeType::FLOATS);
        attributes
    }
}

impl From<f32> for onnx::AttributeProto {
    fn from(value: f32) -> Self {
        let mut attributes = crate::onnx::AttributeProto::new();
        attributes.set_f(value);
        attributes.set_field_type(AttributeProto_AttributeType::FLOAT);
        attributes
    }
}

impl From<i64> for onnx::AttributeProto {
    fn from(value: i64) -> Self {
        let mut attributes = crate::onnx::AttributeProto::new();
        attributes.set_i(value);
        attributes.set_field_type(AttributeProto_AttributeType::INT);
        attributes
    }
}

impl From<String> for onnx::AttributeProto {
    fn from(value: String) -> Self {
        let mut attributes = crate::onnx::AttributeProto::new();
        attributes.set_s(value.into_bytes());
        attributes.set_field_type(AttributeProto_AttributeType::STRING);
        attributes
    }
}

impl From<&str> for onnx::AttributeProto {
    fn from(value: &str) -> Self {
        let mut attributes = crate::onnx::AttributeProto::new();
        attributes.set_s(value.to_string().into_bytes());
        attributes.set_field_type(AttributeProto_AttributeType::STRING);
        attributes
    }
}

impl From<TensorProto> for onnx::AttributeProto {
    fn from(value: TensorProto) -> Self {
        let mut attributes = crate::onnx::AttributeProto::new();
        attributes.set_t(value);
        attributes.set_field_type(AttributeProto_AttributeType::TENSOR);
        attributes
    }
}

// Attribute to value conversions
impl From<&onnx::AttributeProto> for Vec<i64> {
    fn from(value: &onnx::AttributeProto) -> Self {
        value.get_ints().to_vec()
    }
}

impl From<&onnx::AttributeProto> for TensorProto {
    fn from(value: &onnx::AttributeProto) -> Self {
        value.get_t().clone()
    }
}

impl<'a> From<&'a onnx::AttributeProto> for &'a TensorProto {
    fn from(value: &'a onnx::AttributeProto) -> Self {
        value.get_t()
    }
}

impl From<&onnx::AttributeProto> for Vec<f32> {
    fn from(value: &onnx::AttributeProto) -> Self {
        value.get_floats().to_vec()
    }
}

impl From<&onnx::AttributeProto> for f32 {
    fn from(value: &onnx::AttributeProto) -> Self {
        value.get_f()
    }
}

impl From<&onnx::AttributeProto> for i64 {
    fn from(value: &onnx::AttributeProto) -> Self {
        value.get_i()
    }
}

impl From<&onnx::AttributeProto> for String {
    fn from(value: &onnx::AttributeProto) -> Self {
        from_utf8(value.get_s()).unwrap().to_string()
    }
}

#[derive(Error, Debug)]
pub enum OpsetError {
    #[error("more than one ONNX opset was specified: {0} and {1}")]
    DuplicateOnnxOpset(i64, i64),

    #[error("the model references an unknown opset: '{0}'")]
    UnknownOpset(String),
}

pub fn get_opset_version(model: &ModelProto) -> Result<Option<i64>, OpsetError> {
    // Find the version of the ONNX operator set this model is using (this is useful because some operators' specifications change over time).
    // Note, if any other op set than the ONNX operator set is referenced, we cannot run the model.
    // See https://github.com/onnx/onnx/blob/master/docs/Versioning.md#operator-sets
    let mut onnx_opset_version = None;
    for opset_import in model.get_opset_import() {
        match opset_import.get_domain() {
            "" => {
                // This is a reference to the ONNX specification op set
                if let Some(onnx_version) = onnx_opset_version {
                    if opset_import.get_version() != onnx_version {
                        return Err(OpsetError::DuplicateOnnxOpset(
                            onnx_version,
                            opset_import.get_version(),
                        ));
                    }
                } else {
                    onnx_opset_version = Some(opset_import.get_version());
                }
            }
            some_other_opset => {
                return Err(OpsetError::UnknownOpset(some_other_opset.to_string()));
            }
        }
    }
    Ok(onnx_opset_version)
}

pub(crate) fn to_tensor<'model>(
    proto: &'model TensorProto,
) -> Result<Tensor<'model>, DataTypeError> {
    let scalar_type = ScalarType::from_onnx_i32(proto.get_data_type())?;
    let dims = proto
        .get_dims()
        .iter()
        .map(|x| *x as usize)
        .collect::<Vec<usize>>();
    log::debug!(
        "creating tensor for ONNX tensor {} shape {}",
        proto.get_name(),
        Shape::from(scalar_type, &dims)
    );

    let tensor_data: TensorData<'model> = match scalar_type {
        ScalarType::F32 => {
            let fd = proto.get_float_data();
            if fd.is_empty() {
                let fd: &[f32] = bytemuck::cast_slice(proto.get_raw_data());
                TensorData::F32(Cow::from(fd))
            } else {
                TensorData::F32(Cow::from(fd))
            }
        }
        ScalarType::I32 => {
            let fd = proto.get_int32_data();
            if fd.is_empty() {
                let fd: &[i32] = bytemuck::cast_slice(proto.get_raw_data());
                TensorData::I32(Cow::from(fd))
            } else {
                TensorData::I32(Cow::from(fd))
            }
        }
        ScalarType::I64 => {
            let fd = proto.get_int64_data();
            if fd.is_empty() {
                let fd: &[i64] = bytemuck::cast_slice(proto.get_raw_data());
                TensorData::I64(Cow::from(fd))
            } else {
                TensorData::I64(Cow::from(fd))
            }
        }
        ScalarType::U8 => TensorData::U8(Cow::from(proto.get_raw_data())),
    };

    Ok(Tensor {
        data: tensor_data,
        dims,
        display_name: proto.get_name().to_string(),
    })
}

impl<'a> From<&'a AttributeProto> for AttributeValue<'a> {
    fn from(value: &'a AttributeProto) -> Self {
        match value.get_field_type() {
            AttributeProto_AttributeType::INT => AttributeValue::I64(value.get_i()),
            AttributeProto_AttributeType::FLOAT => AttributeValue::F32(value.get_f()),
            AttributeProto_AttributeType::INTS => AttributeValue::I64s(Cow::from(value.get_ints())),
            AttributeProto_AttributeType::FLOATS => {
                AttributeValue::F32s(Cow::from(value.get_floats()))
            }
            AttributeProto_AttributeType::STRING => {
                AttributeValue::String(from_utf8(value.get_s()).unwrap().to_string())
            }
            AttributeProto_AttributeType::TENSOR => {
                AttributeValue::Tensor(to_tensor(value.get_t()).unwrap())
            }
            _ => unimplemented!("attribute field type {}", value.get_field_type().value()),
        }
    }
}

impl OperatorDefinition {
    pub fn from(node: &NodeProto, output_shapes: Vec<Shape>) -> OperatorDefinition {
        assert_eq!(node.get_output().len(), output_shapes.len());
        let mut attributes = HashMap::new();
        for attr in node.get_attribute() {
            attributes.insert(
                attr.get_name().to_string(),
                AttributeValue::from(attr).into_static(),
            );
        }

        OperatorDefinition {
            op_type: node.get_op_type().to_string(),
            attributes,
            output_shapes,
            display_name: node.get_output()[0].to_string(),
        }
    }
}

impl<'model> Node<'model> {
    /// Construct part of the intermediate representation tree for the indicated node.
    pub fn from_node<'a>(
        node: Cow<'model, NodeProto>,
        value_shapes: &HashMap<&'model str, Shape>,
        node_definitions_by_output: &'a HashMap<String, Cow<'model, NodeProto>>,
        nodes_by_output_names: &mut HashMap<String, Arc<Node<'model>>>,
    ) -> Result<Arc<Node<'model>>, IrError> {
        for output_name in node.get_output() {
            if nodes_by_output_names.contains_key(output_name) {
                let n = nodes_by_output_names.get(output_name).unwrap();
                return Ok(n.clone());
            }
        }

        let inputs: Result<Vec<Input<'model>>, IrError> = node
            .get_input()
            .iter()
            .map(|input_name: &String| {
                let my_input_name = input_name.clone();

                // An empty input name signifies missing
                if input_name.is_empty() {
                    return Ok(Input {
                        source_node: Arc::new(Node::new(NodeDefinition::Missing, vec![])),
                        output_index: 0,
                    });
                }

                Ok(match node_definitions_by_output.get(&my_input_name) {
                    Some(source_node_proto) => {
                        // The source is another op - continue translating that node
                        Input {
                            source_node: Node::from_node(
                                source_node_proto.clone(),
                                value_shapes,
                                node_definitions_by_output,
                                nodes_by_output_names,
                            )?,
                            output_index: source_node_proto
                                .get_output()
                                .iter()
                                .position(|s| s == input_name)
                                .ok_or_else(|| {
                                    IrError::OutputNodeNotFound(input_name.to_string())
                                })?,
                        }
                    }
                    None => {
                        Input {
                            output_index: 0,
                            // Did we already translate this node?
                            source_node: match nodes_by_output_names.get(input_name) {
                                Some(node) => node.clone(),
                                None => {
                                    return Err(IrError::InputNodeNotFound {
                                        target_node_name: node.get_name().to_string(),
                                        input_name: input_name.clone(),
                                    })
                                }
                            },
                        }
                    }
                })
            })
            .collect();

        // Obtain output shapes
        let mut output_shapes: Vec<Shape> = Vec::with_capacity(node.get_output().len());
        for output_name in node.get_output() {
            if !value_shapes.contains_key(output_name.as_str()) {
                return Err(IrError::OutputNodeNotFound(output_name.to_string()));
            }

            output_shapes.push(value_shapes[&output_name.as_str()].clone());
        }

        let translated = Arc::new(Node {
            definition: NodeDefinition::Operator(OperatorDefinition::from(&node, output_shapes)),
            inputs: inputs?,
        });

        // Register the translated node by all of its output names
        for output_name in node.get_output() {
            nodes_by_output_names.insert(output_name.clone(), translated.clone());
        }

        Ok(translated)
    }

    /// Construct an intermediate representation graph for calculating the output with the specified name.
    pub fn from_model(
        model: &'model ModelProto,
        outputs: Option<&[String]>,
    ) -> Result<Arc<Node<'model>>, IrError> {
        let graph: &'model GraphProto = model.get_graph();

        // Collect value shapes
        let mut value_shapes: HashMap<&'model str, Shape> = HashMap::new();
        for vi in graph.get_value_info() {
            value_shapes.insert(vi.get_name(), vi.get_shape()?);
        }

        for vi in graph.get_output() {
            let output_name = vi.get_name();
            if !output_name.is_empty() {
                value_shapes.insert(output_name, vi.get_shape()?);
            }
        }

        // Sort nodes by output nodes
        let mut node_definitions_by_output = HashMap::<String, Cow<'model, NodeProto>>::new();
        for node in graph.get_node().iter() {
            for output in node.get_output() {
                if !output.is_empty() {
                    node_definitions_by_output.insert(output.to_string(), Cow::Borrowed(node));
                }
            }
        }

        let mut nodes_by_output_name = HashMap::new();

        // Translate initializers
        for initializer in graph.initializer.iter() {
            nodes_by_output_name.insert(
                initializer.get_name().to_string(),
                Arc::new(Node::new(
                    NodeDefinition::Tensor(to_tensor(initializer)?),
                    vec![],
                )),
            );
        }

        // Translate inputs
        for input in model.get_graph().get_input().iter() {
            if !nodes_by_output_name.contains_key(input.get_name()) {
                nodes_by_output_name.insert(
                    input.get_name().to_string(),
                    Arc::new(Node::new(
                        NodeDefinition::Input {
                            name: input.get_name().to_string(),
                            shape: input.get_shape()?,
                        },
                        vec![],
                    )),
                );
            } else {
                log::warn!(
                    "Skipping input definition {}: already defined",
                    input.get_name()
                );
            }
        }

        let output_names: Vec<String> = match outputs {
            Some(outputs) => outputs.to_vec(),
            None => model
                .get_graph()
                .get_output()
                .iter()
                .map(|x| x.get_name().to_string())
                .collect(),
        };

        let output_nodes: Result<Vec<Input<'model>>, IrError> = output_names
            .iter()
            .map(|output_name| {
                let output_node = model
                    .get_graph()
                    .get_node()
                    .iter()
                    .find(|x| -> bool { x.get_output().contains(output_name) })
                    .ok_or_else(|| IrError::OutputNodeNotFound(output_name.clone()))?;

                let source_node = Node::<'model>::from_node(
                    Cow::Borrowed(output_node),
                    &value_shapes,
                    &node_definitions_by_output,
                    &mut nodes_by_output_name,
                )?;

                let output_index = output_node
                    .get_output()
                    .iter()
                    .position(|s| s == output_name)
                    .ok_or_else(|| IrError::OutputNodeNotFound(output_name.clone()))?;

                Ok(Input {
                    source_node,
                    output_index,
                })
            })
            .collect();

        Ok(Arc::new(Node {
            definition: NodeDefinition::Outputs {
                names: output_names,
            },
            inputs: output_nodes?,
        }))
    }
}

/// Support for creating [`Session`] from ONNX model files.
///
/// # Examples
///
/// Basic usage:
///
/// ```ignore
/// let mut session = Session::from_path("path/to/model.onnx").await.unwrap();
/// ```
impl Session {
    // Read an ONNX model from a path and create a session, using default [session config](SessionConfig).
    pub async fn from_path<P: AsRef<Path>>(path: P) -> Result<Session, SessionError> {
        let model = onnx::ModelProto::parse_from_bytes(&std::fs::read(path)?)?;
        Session::from_model(model).await
    }

    // Read an ONNX model from a path and create a session using the specified [session config](SessionConfig).
    pub async fn from_path_with_config<P: AsRef<Path>>(
        path: P,
        config: &SessionConfig,
    ) -> Result<Session, SessionError> {
        let model = onnx::ModelProto::parse_from_bytes(&std::fs::read(path)?)?;
        Session::from_model_with_config(model, config).await
    }

    /// Read an ONNX model from bytes and create a session, using default [session config](SessionConfig).
    pub async fn from_bytes(bytes: &[u8]) -> Result<Session, SessionError> {
        let model = onnx::ModelProto::parse_from_bytes(bytes)?;
        Session::from_model(model).await
    }

    /// Read an ONNX model from bytes and create a session with the specified [session config](SessionConfig).
    pub async fn from_bytes_with_config(
        bytes: &[u8],
        config: &SessionConfig,
    ) -> Result<Session, SessionError> {
        let model = onnx::ModelProto::parse_from_bytes(bytes)?;
        Session::from_model_with_config(model, config).await
    }

    /// Create a session using the provided [`onnx::ModelProto`] and [session config](SessionConfig).
    pub async fn from_model_with_config(
        model: onnx::ModelProto,
        config: &SessionConfig,
    ) -> Result<Session, SessionError> {
        let (device, queue) = request_device_queue().await;

        // Optimize and compile the model graph to a set of buffers and 'builders' which can basically run GPU shader code referencing these buffers
        let onnx_opset_version = get_opset_version(&model)
            .map_err(SessionError::OpsetError)?
            .ok_or(SessionError::UnknownOnnxOpsetVersion)?;

        let mut optimizer = Optimizer::new(onnx_opset_version);
        let ir = optimizer
            .optimize(Node::from_model(&model, config.outputs.as_deref())?)
            .await?;
        let gpu_model = GpuModel::from(ir, device, queue, onnx_opset_version)?;

        Ok(Session { gpu_model })
    }

    /// Create a Session given an ONNX model, using default configuration.
    pub async fn from_model(model: onnx::ModelProto) -> Result<Session, SessionError> {
        Self::from_model_with_config(model, &SessionConfig::new()).await
    }
}

#[cfg(test)]
mod tests {
    use crate::onnx_model::{
        onnx_attribute, onnx_graph, onnx_initializer, onnx_model, onnx_node, onnx_tensor,
    };
    use crate::tensor::TensorData;

    #[test]
    fn test_use_onnx_model() {
        // USER INPUT
        let n = 5;
        let c = 1;
        let mut input_data = std::collections::HashMap::new();

        let data: Vec<f32> = (0..25).map(|x| x as f32).collect();
        input_data.insert("X".to_string(), data.as_slice().into());

        // ONNX INPUTS
        let shape = vec![1, c, n, n];
        let kernel_n = 3;
        let m = 1;
        let data_w: Vec<f32> = (0..m * c * kernel_n * kernel_n).map(|_| 1.0f32).collect();
        let conv_model = onnx_model(onnx_graph(
            vec![onnx_tensor("X", &shape)],
            vec![onnx_tensor("Y", &[1, 1, 3, 3])],
            vec![],
            vec![onnx_initializer("W", data_w, vec![m, c, 3, 3])],
            vec![onnx_node(
                vec!["X", "W"],
                vec!["Y"],
                "Conv",
                vec![onnx_attribute("kernel_shape", vec![3, 3])],
            )],
        ));

        // LOGIC

        let session = pollster::block_on(crate::Session::from_model(conv_model))
            .expect("Session did not create");

        let result = pollster::block_on(session.run(&input_data)).unwrap();

        assert_eq!(
            result["Y"],
            TensorData::F32(vec![54., 63., 72., 99., 108., 117., 144., 153., 162.].into())
        );
    }
}
