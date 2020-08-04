# SPDX-License-Identifier: MIT
# Copyright © 2020 Patrick Levin
"""Public API of the tensorflowjs graph model Converter"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import tensorflow as tf
import numpy

from tensorflowjs.converters.common import ARTIFACT_MODEL_JSON_FILE_NAME
from tensorflowjs.converters.common import ARTIFACT_MODEL_TOPOLOGY_KEY
from tensorflowjs.converters.common import ARTIFACT_WEIGHTS_MANIFEST_KEY
from tensorflowjs.converters.common import SIGNATURE_KEY
from tensorflowjs.converters.common import USER_DEFINED_METADATA_KEY

from tensorflowjs.read_weights import read_weights
from google.protobuf.json_format import ParseDict

import tfjs_graph_converter.common as common
from tfjs_graph_converter.convert_prelu import replace_prelu, split_fused_prelu
from tfjs_graph_converter.graph_rewrite_util import validate_supported_ops
from tfjs_graph_converter.optimization import optimize_graph
import tfjs_graph_converter.quirks as quirks
import tfjs_graph_converter.util as util

GraphDef = tf.compat.v1.GraphDef
Tensor = numpy.ndarray

SIGNATURE_OUTPUTS = 'outputs'
SIGNATURE_METHOD = 'method_name'


# this class exists as a way to ensure valid mappings without having check
# inside consumers of the type
class RenameMap:
    """Mapping from input and output nodes to new names"""
    def __init__(self, mapping):
        """Set input and output remapping

            Args:
                mapping: mapping from old to new names - any type that
                         can be converted to a Dict[str, str].
                         New names must be unique.
        """
        def _is_valid_str(s): return isinstance(s, str) and len(s.strip()) > 0
        def _is_valid(k, v): return _is_valid_str(k) and _is_valid_str(v)
        if not isinstance(mapping, dict):
            try:
                mapping = dict(mapping)
            except TypeError as t:
                raise ValueError(f'{t}')
        if any((k, v) for k, v in mapping.items() if not _is_valid(k, v)):
            raise ValueError('mapping must be between non-empty strings')
        self.mapping = mapping

    def apply(self, signature: util.SignatureDef) -> util.SignatureDef:
        """Apply the rename mapping to a SignatureDef"""
        if not isinstance(signature, util.SignatureDef):
            raise ValueError('signature must be a SignatureDef proto')

        # replace the contents of a mapfield with another mapfield
        def _replace_contents(target, replacement):
            keys = [key for key in target]
            # clear the target
            for key in keys:
                del target[key]
            # copy the contents
            for key in replacement:
                target[key].CopyFrom(replacement[key])

        updated = signature
        if any(self.mapping):
            # operate on a local copy such that the order of replacements
            # doesn't matter (i.e. allow for A=B & B=A name swapping)
            inputs, outputs = signature.inputs, signature.outputs
            temp = util.SignatureDef(inputs=inputs, outputs=outputs)
            for old_name, new_name in self.mapping.items():
                if old_name in inputs:
                    temp.inputs[new_name].CopyFrom(inputs[old_name])
                    del temp.inputs[old_name]
                elif old_name in outputs:
                    temp.outputs[new_name].CopyFrom(outputs[old_name])
                    del temp.outputs[old_name]
            _replace_contents(inputs, temp.inputs)
            _replace_contents(outputs, temp.outputs)
        return updated


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


def _extract_signature_def(model_json: Dict[str, Any]
                           ) -> Optional[util.SignatureDef]:
    """
    Extract the signature definition from the model's meta data.

    Args:
        model_json: JSON dict from TFJS model file

    Returns:
        TF SignatureDef proto; None if meta data is missing or incomplete
    """
    # three possible scenarios:
    #   1. meta data contains a valid signature w/ inputs and outputs
    #   2. meta data contains incomplete signature (missing in- or outputs)
    #   3. meta data is missing or doesn't contain signature
    # this function works for scenario 1)
    if USER_DEFINED_METADATA_KEY not in model_json:
        return None
    meta_data = model_json[USER_DEFINED_METADATA_KEY]
    if SIGNATURE_KEY not in meta_data:
        return None
    signature = meta_data[SIGNATURE_KEY]
    if tf.saved_model.PREDICT_INPUTS not in signature:
        return None
    if tf.saved_model.PREDICT_OUTPUTS not in signature:
        return None
    signature_def = ParseDict(signature, util.SignatureDef())
    if len(signature_def.method_name) == 0:
        signature_def.method_name = tf.saved_model.PREDICT_METHOD_NAME

    def _remove_channel_from_key(mapfield):
        names_to_fix = [key for key in mapfield if key.endswith(':0')]
        for name in names_to_fix:
            mapfield[name[0:-2]].CopyFrom(mapfield[name])
            del mapfield[name]

    _remove_channel_from_key(signature_def.inputs)
    _remove_channel_from_key(signature_def.outputs)
    return signature_def


def _create_graph(graph_def: GraphDef,
                  weight_dict: Dict[str, Tensor],
                  modifiers: Dict[str, Callable]) -> tf.Graph:
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

    graph_def = optimize_graph(graph)
    return graph_def_to_graph_v1(graph_def)


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
                                  base_path: str
                                  ) -> Tuple[tf.Graph, util.SignatureDef]:
    """
    Convert TFJS JSON model to TF Graph

    Args:
        model_json: JSON dict from TFJS model file
        base_path:  Path to the model file (where to find the model weights)

    Returns:
        Tuple of TF Graph for inference or saving and TF signature definition
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

    graph = _create_graph(graph_def, weight_dict, weight_modifiers)
    signature_def = _extract_signature_def(model_json) or util.infer_signature(
        graph)
    return (graph, signature_def)


def load_graph_model_and_signature(model_dir: str
                                   ) -> Tuple[tf.Graph, util.SignatureDef]:
    """
    Load a TFJS Graph Model from a directory

    Args:
        model_dir: Directory that contains the tfjs model.json and weights;
                alternatively name and path of the model.json if the name
                differs from the default ("model.json")

    Returns:
        Tupel of TF frozen graph for inference or saving and TF signature def
    """
    model_path, model_name = _parse_path_and_model_json(model_dir)
    model_file_path = os.path.join(model_path, model_name)
    with open(model_file_path, "r") as model_file:
        model_json = json.load(model_file)
    return _convert_graph_model_to_graph(model_json, model_path)


def load_graph_model(model_dir: str) -> tf.Graph:
    """
    Load a TFJS Graph Model from a directory

    Args:
        model_dir: Directory that contains the tfjs model.json and weights;
                alternatively name and path of the model.json if the name
                differs from the default ("model.json")

    Returns:
        TF frozen graph for inference or saving
    """
    graph, _ = load_graph_model_and_signature(model_dir)
    return graph


def graph_def_to_graph_v1(graph_def: GraphDef) -> tf.Graph:
    """Convert a GraphDef protobuf message to a tf.Graph

    Use this function to convert the graph message returned by
    `load_graph_model` to a tf.Graph that can be used for inference.

    Args:
        graph_def: GraphDef protobuf message as returned by `load_graph_model`

    Returns:
        tf.Graph for inference.
    """
    graph = tf.Graph()
    with tf.compat.v1.Session(graph=graph):
        tf.graph_util.import_graph_def(graph_def, name='')
    return graph


def graph_to_function_v2(graph: Union[GraphDef, tf.Graph]) -> Callable:
    """Wrap a GraphDef or TF1 frozen graph in a TF2 function for easy inference

    Use this function to convert a GraphDef returned by `load_graph_model` or
    a TF v1 frozen graph into a callable TF2 function.

    Args:
        graph: GraphDef protocol buffer message or TF1 frozen graph

    Returns:
        The function returns a TF2 wrapped function that is callable with
        input tensors or `numpy` arrays as arguments and returns a list of
        model outputs as tensors.
    """
    graph_def = graph.as_graph_def() if isinstance(graph, tf.Graph) else graph

    def _imports_graph_def():
        tf.graph_util.import_graph_def(graph_def, name='')

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph
    inputs = util.get_input_tensors(graph_def)
    outputs = util.get_output_tensors(graph_def)
    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))


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
                               tags: List[str] = None,
                               signature_def_map: dict = None,
                               signature_key_map: RenameMap = None) -> str:
    """
    Convert a TFJS graph model to a SavedModel

    Args:
        model_dir: Directory that contains the TFJS JSON model and weights
        export_dir: Target directory to save the TF model in
        tags: Tags for the SavedModel, optional
        signature_def_map: Dictionary that maps signature keys to model
            signatures;
            A model signature is a dictionary that maps a signature key
            (the name of the signature) - which can be empty or `None` - to
            a dictionary that contains two keys:
            - `outputs`, which maps to a flat list of output tensor names
            - `method_name`, which is an optional name describing the signature
              (defaults to `tf.saved_model.PREDICT_METHOD_NAME`)

            Example:
            {'model_signature': {'outputs': ['Output1', 'Output2']},
            {'debug_signature': {'outputs': ['Output1', 'Output2', 'Detail']}}}

            The names given in the 'outputs' must refer to output tensors in
            the model; otherwise a `ValueError` is raised.
        signature_key_map: Mapping from input- or output tensors to signature
                           keys. The default signature uses tensor names for
                           signature keys. This argument allows to map tensor
                           names to different keys.

    Returns:
        The path to which the model was written.
    """
    graph, signature_def = load_graph_model_and_signature(model_dir)
    builder = tf.compat.v1.saved_model.Builder(export_dir)
    signature_map = _get_signature_map(graph, signature_def, signature_def_map)
    tags = _get_tags(tags)
    if signature_key_map is not None:
        for signature in signature_map.values():
            signature_key_map.apply(signature)

    with tf.compat.v1.Session(graph=graph) as sess:
        builder.add_meta_graph_and_variables(sess, tags=tags,
                                             signature_def_map=signature_map)
    return builder.save()


def graph_models_to_saved_model(model_list: List[Tuple[str, List[str]]],
                                export_dir: str,
                                signatures: dict = None,
                                signature_keys: Dict[str, RenameMap] = None
                                ) -> str:
    """
    Read multiple TFJS graph models and saves them in a single SavedModel

    Args:
        model_list: List of tuples containing TFJS model dir and tags, e.g.
            [("./models/model1", ["step1"]), ("./models/model2": ["step2"])]
        export_dir: Target directory to save the TF model in
        signatures: Dictionary of model names (i.e. the first item in
            `model_list` tuples) to per-model signatures.
            A model signature is a dictionary that maps a signature key
            (the name of the signature) - which can be empty or `None` - to
            a dictionary that contains two keys:
            - `outputs`, which maps to a flat list of output tensor names
            - `method_name`, which is an optional name describing the signature
              (defaults to `tf.saved_model.PREDICT_METHOD_NAME`)
            Empty keys map to the default signature key.
            Keys that don't map to a model in ``model_list`` are ignored.

            {'./models/model2': {'': {'outputs': ['model_output1']},
                                 'debug': {'outputs': ['model_output1',
                                                       'details']}}

            If ``signatures`` is not provided or doesn't contain a key matching
            a model in ``model_list``, the default signature is used.
        signature_keys: Dictionary of model names (i.e. the first item in
            `model_list` tuples) to per-model signature key mappings. This
            allows a remapping of signature inputs and outputs to different
            keys (the tensor names stay unaffected).

    Returns:
        The path to which the model was written.
    """
    signatures = signatures or {}
    signature_keys = signature_keys or {}
    builder = tf.compat.v1.saved_model.Builder(export_dir)

    def _apply_key_map(model, signature_map):
        if model in signature_keys and signature_keys[model] is not None:
            signature_key_map = signature_keys[model]
            for signature in signature_map.values():
                signature_key_map.apply(signature)

    model_dir, tags = _get_tags(model_list[0])
    graph, signature_def = load_graph_model_and_signature(model_dir)
    signature = signatures[model_dir] if model_dir in signatures else None
    signature_map = _get_signature_map(graph, signature_def, signature)
    _apply_key_map(model_dir, signature_map)

    with tf.compat.v1.Session(graph=graph) as sess:
        builder.add_meta_graph_and_variables(sess, tags=tags,
                                             signature_def_map=signature_map)

    for model_dir, tags in model_list[1:]:
        graph, signature_def = load_graph_model_and_signature(model_dir)
        signature = signatures[model_dir] if model_dir in signatures else None
        signature_map = _get_signature_map(graph, signature_def, signature)
        _apply_key_map(model_dir, signature_map)
        tags = _get_tags(tags)
        with tf.compat.v1.Session(graph=graph):
            builder.add_meta_graph(tags=tags, signature_def_map=signature_map)

    return builder.save()


def _get_signature_map(graph: tf.Graph, default_signature_def: dict,
                       signature_def_map: dict) -> dict:
    """Select the signature def map from default & user provided signatures"""
    default_key = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    if signature_def_map is None:
        return {default_key: default_signature_def}
    else:
        return _build_signatures(graph, signature_def_map)


def _build_signatures(graph: tf.Graph, signature_def_map: dict) -> dict:
    """
    Turn a signature map into a proper argument for ``add_meta_graph`` by
    creating tensor info for each output name given in ``signature_def_map``.

    Args:
        graph: TF Graph instance
        signature_def_map: Dictionary that maps signature keys to output names

    Returns:
        Signature definition map for passing to
        ``SavedModelBuilder.add_meta_graph``
    """
    inputs = {info.name: util._build_tensor_info(graph, info)
              for info in util.get_input_nodes(graph)}
    default_key = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    output_tensors = set(name[0:-2] for name in util.get_output_tensors(graph))
    for key, value in signature_def_map.items():
        if SIGNATURE_OUTPUTS not in value:
            raise ValueError(f'Signature "{key or default_key}" is invalid: '
                             f'the key "{SIGNATURE_OUTPUTS}" is missing')
        if len(value[SIGNATURE_OUTPUTS]) == 0:
            raise ValueError(f'Signature key "{SIGNATURE_OUTPUTS}" must not'
                             'be empty')
        for name in value[SIGNATURE_OUTPUTS]:
            if name not in output_tensors:
                valid_outputs = str(output_tensors)[1:-1]
                raise ValueError(f'Signature "{key or default_key}" is '
                                 f'invalid: "{name}" is not an output tensor.'
                                 f' Valid outputs are {valid_outputs}')
    return {key or default_key: _build_signature(graph, inputs, info)
            for key, info in signature_def_map.items()}


def _build_signature(graph: tf.Graph, inputs: dict,
                     info: dict) -> util.SignatureDef:
    """
    Return tensor info for inputs and outputs.

    Args:
        graph: TF Graph instance
        inputs: dict that maps input names to tensor information
        info: dict containing the key 'outputs' that maps to flat list of
            output names and an optional key 'method_name'

    Returns:
        dict that maps inputs and outputs to tensor information
        (name, type, and shape)
    """
    def _info(name):
        return util.NodeInfo(name=name, shape=(), dtype=0, tensor=name+':0')

    output_names = info[SIGNATURE_OUTPUTS]
    outputs = {name: util._build_tensor_info(graph, _info(name))
               for name in output_names}
    name = tf.saved_model.PREDICT_METHOD_NAME
    if SIGNATURE_METHOD in info:
        name = info[SIGNATURE_METHOD] or name
    return util.SignatureDef(inputs=inputs, outputs=outputs, method_name=name)


def _get_tags(tags):
    """Reformat tags to a proper list of trimmed strings"""
    if tags is None:
        tags = tf.saved_model.SERVING
    if isinstance(tags, str):
        tags = [tags]
    # remove all empty and whitespace-only tags and return trimmed strings
    return [tag.strip() for tag in tags if bool(tag) and len(tag.strip()) > 0]
