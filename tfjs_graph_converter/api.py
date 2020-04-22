# SPDX-License-Identifier: MIT
# Copyright © 2020 Patrick Levin
"""Public API of the tensorflowjs graph model Converter"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import os

import tensorflow as tf
import tensorflowjs.converters.common as tfjs_common

from tensorflowjs.read_weights import read_weights
from google.protobuf.json_format import ParseDict

import tfjs_graph_converter.common as common
import tfjs_graph_converter.quirks as quirks
from tfjs_graph_converter.graph_rewrite_util import get_op_def


def _parse_path_and_model_json(model_dir):
    """
    Parse model directory name and return path and file name

    Args:
        model_dir: Model file path - either directory name or path + file name

    Returns:
        Tuple of directory name and model file name (without directory)
    """
    if model_dir.endswith('.json'):
        if not os.path.isfile(model_dir):
            raise ValueError(f'Model not found: {model_dir}')
        return os.path.split(model_dir)
    if os.path.isdir(model_dir):
        return model_dir, tfjs_common.ARTIFACT_MODEL_JSON_FILE_NAME
    raise ValueError(f'Model path is not a directory: {model_dir}')


def _get_op_list(message_dict):
    """
    Return the set of operations used in the graph.

    Args:
        message_dict: deserialised JSON message

    Returns:
        Set of operations in the graph
    """
    ops = set()
    if common.TFJS_NODE_KEY in message_dict:
        nodes = message_dict[common.TFJS_NODE_KEY]
        ops = set(node[common.TFJS_OP_KEY] for node in nodes)
    return ops


def _verify_supported_ops(op_list):
    """
    Verify supported operations and raise an error if the graph
    contains unsupported layers.

    Args:
        op_list: Iterable of operation names (strings) contained in the graph
    """
    for op_name in op_list:
        if get_op_def(op_name) is None:
            raise ValueError(f'Unsupported operation: "{op_name}"')


def _convert_graph_def(message_dict):
    """
    Convert JSON to TF GraphDef message

    Args:
        message_dict: deserialised JSON message

    Returns:
        TF GraphDef message
    """
    message_dict = quirks.fix_node_attributes(message_dict)
    return ParseDict(message_dict, tf.compat.v1.GraphDef())


def _create_graph(graph_def, weight_dict, op_list):
    """
    Create a TF Graph from nodes

    Args:
        graph_def: TF GraphDef message containing the node graph
        weight_dict: Dictionary from node names to tensor data
        op_list: Set of operations in the graph

    Returns:
        TF Graph for inference or saving
    """
    graph = tf.Graph()
    _verify_supported_ops(op_list)
    with tf.compat.v1.Session(graph=graph):
        for key, value in weight_dict.items():
            weight_dict[key] = tf.convert_to_tensor(value)
        tf.graph_util.import_graph_def(graph_def, weight_dict, name='')

    return graph


def _convert_graph_model_to_graph(model_json, base_path):
    """
    Convert TFJS JSON model to TF Graph

    Args:
        model_json: JSON dict from TFJS model file
        base_path:  Path to the model file (where to find the model weights)

    Returns:
        TF Graph for inference or saving
    """
    if tfjs_common.ARTIFACT_MODEL_TOPOLOGY_KEY not in model_json:
        raise ValueError("model_json is missing key '{}'".format(
            tfjs_common.ARTIFACT_MODEL_TOPOLOGY_KEY))

    topology = model_json[tfjs_common.ARTIFACT_MODEL_TOPOLOGY_KEY]

    if tfjs_common.ARTIFACT_WEIGHTS_MANIFEST_KEY not in model_json:
        raise ValueError("model_json is missing key '{}'".format(
            tfjs_common.ARTIFACT_WEIGHTS_MANIFEST_KEY))

    weights_manifest = model_json[tfjs_common.ARTIFACT_WEIGHTS_MANIFEST_KEY]
    weight_list = read_weights(weights_manifest, base_path, flatten=True)

    op_list = _get_op_list(topology)
    graph_def = _convert_graph_def(topology)
    name, data = common.TFJS_NAME_KEY, common.TFJS_DATA_KEY
    weight_dict = dict((weight[name], weight[data]) for weight in weight_list)

    return _create_graph(graph_def, weight_dict, op_list)


def load_graph_model(model_dir):
    """
    Load a TFJS Graph Model from a directory

    Args:
        model_dir: Directory that contains the tfjs model.json and weights;
                alternatively name and path of the model.json if the name
                differs from the default ("model.json")

    Returns:
        TF frozen graph for inference or saving
    """
    model_path, model_name = _parse_path_and_model_json(model_dir)
    model_file_path = os.path.join(model_path, model_name)
    with open(model_file_path, "r") as model_file:
        model_json = json.load(model_file)

    return _convert_graph_model_to_graph(model_json, model_path)


def graph_model_to_frozen_graph(model_dir, export_path):
    """
    Convert a TFJS graph model to a frozen TF graph

    Args:
        model_dir: Directory that contains the TFJS JSON model and weights
        export_path: Path to the frozen graph (e.g. './output.pb')
    """
    export_dir = os.path.dirname(export_path)
    model_name = os.path.basename(export_path)

    graph = load_graph_model(model_dir)
    return tf.io.write_graph(graph, export_dir, model_name, as_text=False)


def graph_model_to_saved_model(model_dir, export_dir, tags):
    """
    Convert a TFJS graph model to a SavedModel

    Args:
        model_dir: Directory that contains the TFJS JSON model and weights
        export_dir: Target directory to save the TF model in
        tags: Tags for the SavedModel
    """
    graph = load_graph_model(model_dir)
    builder = tf.compat.v1.saved_model.Builder(export_dir)

    with tf.compat.v1.Session(graph=graph) as sess:
        builder.add_meta_graph_and_variables(sess, tags=tags)
    return builder.save()


def graph_models_to_saved_model(model_list, export_dir):
    """
    Reads multiple TFJS graph models and saves them in a single SavedModel

    Args:
        model_list: List of tuples containing TFJS model dir and tags, e.g.
            [("./models/model1", ["step1"]), ("./models/model2": ["step2"])]
        export_dir: Target directory to save the TF model in
    """
    builder = tf.compat.v1.saved_model.Builder(export_dir)

    model_dir, tags = model_list[0]
    graph = load_graph_model(model_dir)
    with tf.compat.v1.Session(graph=graph) as sess:
        builder.add_meta_graph_and_variables(sess, tags=tags)

    for model_dir, tags in model_list[1:]:
        graph = load_graph_model(model_dir)
        with tf.compat.v1.Session(graph=graph):
            builder.add_meta_graph(tags=tags)

    return builder.save()
