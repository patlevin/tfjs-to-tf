# SPDX-License-Identifier: MIT
# Copyright © 2020 Patrick Levin
"""Public API of the tensorflowjs graph model Converter"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import os
from typing import Any, Callable, Dict, List, Tuple

import tensorflow as tf
import numpy

from tensorflowjs.converters.common import ARTIFACT_MODEL_JSON_FILE_NAME
from tensorflowjs.converters.common import ARTIFACT_MODEL_TOPOLOGY_KEY
from tensorflowjs.converters.common import ARTIFACT_WEIGHTS_MANIFEST_KEY

from tensorflowjs.read_weights import read_weights
from google.protobuf.json_format import ParseDict

import tfjs_graph_converter.common as common
from tfjs_graph_converter.convert_prelu import replace_prelu, split_fused_prelu
from tfjs_graph_converter.graph_rewrite_util import validate_supported_ops
from tfjs_graph_converter.optimization import optimize_graph
import tfjs_graph_converter.quirks as quirks

GraphDef = tf.compat.v1.GraphDef
Tensor = numpy.ndarray


def _parse_path_and_model_json(model_dir: str) -> Tuple[str, str]:
    """
    Parse model directory name and return path and file name

    Args:
        model_dir: Model file path - either directory name or path + file name

    Raises:
        ValueError: Model or model directory don't exist

    Returns:
        Tuple of directory name and model file name (without directory)
    """
    if model_dir.endswith('.json'):
        if not os.path.isfile(model_dir):
            raise ValueError(f'Model not found: {model_dir}')
        return os.path.split(model_dir)
    if os.path.isdir(model_dir):
        return model_dir, ARTIFACT_MODEL_JSON_FILE_NAME
    raise ValueError(f'Model path is not a directory: {model_dir}')


def _convert_graph_def(message_dict: Dict[str, Any]) -> GraphDef:
    """
    Convert JSON to TF GraphDef message

    Args:
        message_dict: deserialised JSON message

    Returns:
        TF GraphDef message
    """
    message_dict = quirks.fix_node_attributes(message_dict)
    return ParseDict(message_dict, tf.compat.v1.GraphDef())


def _create_graph(graph_def: GraphDef,
                  weight_dict: Dict[str, Tensor],
                  modifiers: Dict[str, Callable]) -> GraphDef:
    """
    Create a TF Graph from nodes

    Args:
        graph_def: TF GraphDef message containing the node graph
        weight_dict: Dictionary from node names to tensor data
        modifiers: Operations to be performed on weights before the conversion

    Raises:
        ValueError: The given graph def contains unsupported operations

    Returns:
        TF Graph for inference or saving
    """
    graph = tf.Graph()
    validate_supported_ops(graph_def)
    with tf.compat.v1.Session(graph=graph):
        for key, value in weight_dict.items():
            if key in modifiers:
                value = (modifiers[key])(value)
            weight_dict[key] = tf.convert_to_tensor(value)
        tf.graph_util.import_graph_def(graph_def, weight_dict, name='')

    optimised_graph = optimize_graph(graph)
    return optimised_graph


def _replace_unsupported_operations(
        input_graph_def: GraphDef) -> Tuple[GraphDef, Dict[str, Callable]]:
    """Replace known unsupported operations by rewriting the input graph"""
    weight_modifiers = dict()
    # split fused ops that contain unsupported activations
    new_graph, modifiers = split_fused_prelu(input_graph_def)
    weight_modifiers.update(modifiers)
    # replace unsupported activations
    new_graph, modifiers = replace_prelu(new_graph)
    weight_modifiers.update(modifiers)
    return new_graph, weight_modifiers


def _convert_graph_model_to_graph(model_json: Dict[str, Any],
                                  base_path: str) -> GraphDef:
    """
    Convert TFJS JSON model to TF Graph

    Args:
        model_json: JSON dict from TFJS model file
        base_path:  Path to the model file (where to find the model weights)

    Returns:
        TF Graph for inference or saving
    """
    if ARTIFACT_MODEL_TOPOLOGY_KEY not in model_json:
        raise ValueError(
            f"model_json is missing key '{ARTIFACT_MODEL_TOPOLOGY_KEY}'")

    topology = model_json[ARTIFACT_MODEL_TOPOLOGY_KEY]

    if ARTIFACT_WEIGHTS_MANIFEST_KEY not in model_json:
        raise ValueError(f'{ARTIFACT_MODEL_JSON_FILE_NAME} is missing key '
                         f"'{ARTIFACT_WEIGHTS_MANIFEST_KEY}'")

    weights_manifest = model_json[ARTIFACT_WEIGHTS_MANIFEST_KEY]
    weight_list = read_weights(weights_manifest, base_path, flatten=True)

    graph_def = _convert_graph_def(topology)
    name, data = common.TFJS_NAME_KEY, common.TFJS_DATA_KEY
    weight_dict = dict((weight[name], weight[data]) for weight in weight_list)
    graph_def, weight_modifiers = _replace_unsupported_operations(graph_def)

    return _create_graph(graph_def, weight_dict, weight_modifiers)


def load_graph_model(model_dir: str) -> GraphDef:
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


def graph_model_to_frozen_graph(model_dir: str, export_path: str) -> str:
    """
    Convert a TFJS graph model to a frozen TF graph

    Args:
        model_dir: Directory that contains the TFJS JSON model and weights
        export_path: Path to the frozen graph (e.g. './output.pb')

    Returns:
        The path to the output proto-file.
    """
    export_dir = os.path.dirname(export_path)
    model_name = os.path.basename(export_path)

    graph = load_graph_model(model_dir)
    return tf.io.write_graph(graph, export_dir, model_name, as_text=False)


def graph_model_to_saved_model(model_dir: str,
                               export_dir: str,
                               tags: List[str]) -> str:
    """
    Convert a TFJS graph model to a SavedModel

    Args:
        model_dir: Directory that contains the TFJS JSON model and weights
        export_dir: Target directory to save the TF model in
        tags: Tags for the SavedModel

    Returns:
        The path to which the model was written.
    """
    graph = load_graph_model(model_dir)
    builder = tf.compat.v1.saved_model.Builder(export_dir)

    with tf.compat.v1.Session(graph=graph) as sess:
        builder.add_meta_graph_and_variables(sess, tags=tags)
    return builder.save()


def graph_models_to_saved_model(model_list: List[Tuple[str, List[str]]],
                                export_dir: str) -> str:
    """
    Read multiple TFJS graph models and saves them in a single SavedModel

    Args:
        model_list: List of tuples containing TFJS model dir and tags, e.g.
            [("./models/model1", ["step1"]), ("./models/model2": ["step2"])]
        export_dir: Target directory to save the TF model in

    Returns:
        The path to which the model was written.
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
