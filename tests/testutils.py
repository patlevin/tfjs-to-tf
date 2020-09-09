# SPDX-License-Identifier: MIT
# Copyright © 2020 Patrick Levin
"""Unit test utilities"""
import json
import os
from typing import Any, Callable, Dict, List, Optional, Union

import numpy
import tensorflow as tf
import tensorflow.python.framework.convert_to_constants as cc
from google.protobuf.json_format import MessageToJson, ParseDict

from tfjs_graph_converter.graph_rewrite_util import GraphDef, NodeDef

Tensor = numpy.ndarray
SAMPLE_MODEL_FILE_NAME = './models/atari_net.json'
SIMPLE_MODEL_PATH_NAME = './models/simple/'
SIMPLE_MODEL_FILE_NAME = './models/simple/frozen_graph.pb'
PRELU_MODEL_PATH = './models/prelu/'
PRELU_MODEL_FILE = './models/prelu/keras.h5'
MULTI_HEAD_PATH = './models/multi_head/'
MULTI_HEAD_FILE = './models/multi_head/frozen_graph.pb'
DEPTHWISE_RELU_PATH = './models/depthwise_relu/'
DEPTHWISE_RELU_FILE = './models/depthwise_relu/frozen_graph.pb'
DEPTHWISE_PRELU_PATH = './models/depthwise_prelu/'
DEPTHWISE_PRELU_FILE = './models/depthwise_prelu/frozen_graph.pb'
IMAGE_DATASET = './data/horses-and-humans.zip'


def get_inputs(graph_def: GraphDef) -> List[NodeDef]:
    """Return all input nodes from a graph"""
    def is_input(node):
        return node.op == 'Placeholder' and len(node.input) == 0
    return [node for node in graph_def.node if is_input(node)]


def get_outputs(graph_def: GraphDef) -> List[NodeDef]:
    """Return all output nodes from a graph"""
    def is_op_node(node):
        return node.op not in ('Const', 'Placeholder')
    nodes = [node for node in graph_def.node if is_op_node(node)]

    def has_ref(node):
        return any(ref for ref in nodes if node.name in ref.input)
    return [node for node in nodes if not has_ref(node)]


def graph_to_model(graph: Union[tf.Graph, GraphDef, str],
                   weight_dict: Dict[str, Tensor] = {}) -> Callable:
    """Convert a TF v1 frozen graph to a TF v2 function for easy inference"""
    graph_def = graph
    if isinstance(graph, tf.Graph):
        graph_def = graph.as_graph_def()
    elif isinstance(graph, str):
        # graph is a file name: load graph from disk
        if graph.endswith('.json'):
            with open(graph, 'r') as json_file:
                message_dict = json.loads(json_file.read())
            graph_def = ParseDict(message_dict, GraphDef())
        elif graph.endswith('.h5'):
            # Keras model - just load and return as-is
            return tf.keras.models.load_model(graph)
        else:
            with open(graph, 'rb') as proto_file:
                string = proto_file.read()
            graph_def = GraphDef()
            graph_def.ParseFromString(string)

    tensor_dict = dict()

    def _imports_graph_def():
        for name, data in weight_dict.items():
            tensor_dict[name] = tf.convert_to_tensor(data)
        tf.graph_util.import_graph_def(graph_def, tensor_dict, name='')

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph
    inputs = [(node.name+':0') for node in get_inputs(graph_def)]
    outputs = [(node.name+':0') for node in get_outputs(graph_def)]
    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))


def model_to_graph(model: Callable) -> tf.Graph:
    """Convert a keras model into a wrapped function graph"""
    input_specs = [tf.TensorSpec(t.shape, t.dtype) for t in model.inputs]
    wrapped_function = tf.function(lambda x: model(x))
    wrapped_function = wrapped_function.get_concrete_function(input_specs)
    frozen_function = cc.convert_variables_to_constants_v2(wrapped_function)
    return frozen_function.graph


def select_all(dict_obj: Dict[str, Any], key_name: str) -> List[Any]:
    """Select all dictionary values of a given name, even when deeply nested"""
    stack = list(dict_obj.items())
    results = []
    while len(stack) > 0:
        key, value = stack.pop()
        if key == key_name:
            results.append(value)
        elif isinstance(value, dict):
            stack.extend(value.items())
        elif isinstance(value, list):
            for val in value:
                if isinstance(val, dict):
                    stack.extend(val.items())
    results.reverse()   # stack-based approach reverses original order
    return results


def select_single(dict_obj: Dict[str, Any], key_name: str) -> Any:
    """Select the single matching dictionary value of the provided name"""
    matches = select_all(dict_obj, key_name)
    if len(matches) != 1:
        raise ValueError(f"No unique match for '{key_name}'")
    return matches[0]


def node_proto_from_json(node_json: str) -> NodeDef:
    """Return a nodedef protobuf message from a raw JSON string"""
    node_dict = json.loads(node_json)
    node_def = ParseDict(node_dict, NodeDef())
    return node_def


def get_path_to(file_name: str) -> str:
    """Return the path to the given file relative to the script location"""
    file_path = os.path.abspath(__file__)
    path, _ = os.path.split(file_path)
    return os.path.normpath(os.path.join(path, file_name))


def get_sample_graph_def(
                model_name: Optional[str] = None,
                fmt: str = 'proto') -> Union[GraphDef, Dict[str, Any], str]:
    """Return a sample model as protobuf message or JSON dictionary

        Args:
            model_name: optional model file name (relative to script location)
            fmt: "proto" returns a protocol buffer message; "dict" returns
                 a JSON dictionary; "json" returns a JSON string

        Raises:
            ValueError: format is unknown

        Returns:
            Sample model (see generate_test_model.py) as message, dict,
            or raw json
    """
    model_file_name = get_path_to(model_name or SAMPLE_MODEL_FILE_NAME)
    if model_file_name.endswith('.json'):
        with open(model_file_name, 'r') as json_file:
            message = json_file.read()
            if fmt == 'json':
                return message
            message = json.loads(message)
            if fmt == 'dict':
                return message
            if fmt == 'proto':
                return ParseDict(message, GraphDef())
            raise ValueError(f'Unsupported output format: "{fmt}", expected '
                             '"json", "dict", or "proto"')
    else:
        with open(model_file_name, 'rb') as proto_file:
            message = GraphDef()
            message.ParseFromString(proto_file.read())
            if fmt == 'proto':
                return message
            message = MessageToJson(message)
            if fmt == 'json':
                return message
            message = json.loads(message)
            if fmt == 'dict':
                return message
            raise ValueError(f'Unsupported output format: "{fmt}", expected '
                             '"json", "dict", or "proto"')


def get_sample_graph(model_name: Optional[str] = None) -> tf.Graph:
    """Return a sample model as tf.Graph"""
    graph_def = get_sample_graph_def(model_name, fmt='proto')
    graph = tf.Graph()
    with tf.compat.v1.Session(graph=graph):
        tf.graph_util.import_graph_def(graph_def, name='')
    return graph
