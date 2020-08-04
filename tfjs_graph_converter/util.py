# SPDX-License-Identifier: MIT
# Copyright © 2020 Patrick Levin
"""Utility functions for working with TensorFlow Graphs"""
from __future__ import absolute_import

from collections import defaultdict, namedtuple
from typing import List, Optional, Union

import numpy as np
import tensorflow as tf
from tensorflow.core.framework.graph_pb2 import GraphDef
from tensorflow.core.framework.node_def_pb2 import NodeDef
from tensorflow.core.protobuf.meta_graph_pb2 import SignatureDef
from tensorflow.core.protobuf.meta_graph_pb2 import TensorInfo

import tfjs_graph_converter.common as c
import tfjs_graph_converter.graph_rewrite_util as rewrite

_DTYPE_MAP: List[type] = [
    None,
    np.float32,
    np.float64,
    np.int32,
    np.uint8,
    np.int16,
    np.int8,
    None,
    np.complex64,
    np.int64,
    np.bool
]

NodeInfo = namedtuple('NodeInfo', 'name shape dtype tensor')


def _is_op_node(node: NodeInfo) -> bool:
    return node.op not in (c.TFJS_NODE_CONST_KEY, c.TFJS_NODE_PLACEHOLDER_KEY)


def _op_nodes(graph_def: GraphDef) -> List[NodeDef]:
    return [node for node in graph_def.node if _is_op_node(node)]


def _map_type(type_id: int) -> type:
    if type_id < 0 or type_id > len(_DTYPE_MAP):
        raise ValueError(f'Unsupported data type: {type_id}')
    np_type = _DTYPE_MAP[type_id]
    return np_type


def _get_shape(node: NodeDef) -> List[int]:
    def shape(attr): return attr.shape.dim
    def size(dim): return dim.size if dim.size > 0 else None
    # accessing non-existing keys would CREATE the key in mapfields!
    if c.TFJS_ATTR_SHAPE_KEY in node.attr:
        return [size(dim) for dim in shape(node.attr[c.TFJS_ATTR_SHAPE_KEY])]
    else:
        return []


def _dtype(node: NodeDef) -> Optional[int]:
    # accessing non-existing keys would CREATE the key in mapfields!
    if c.TFJS_ATTR_DTYPE_KEY in node.attr:
        return _map_type(node.attr[c.TFJS_ATTR_DTYPE_KEY].type)
    elif 'T' in node.attr:
        return _map_type(node.attr['T'].type)
    else:
        return None


def _node_info(node: NodeDef) -> NodeInfo:
    return NodeInfo(name=node.name, shape=_get_shape(node), dtype=_dtype(node),
                    tensor=node.name + ':0')


def _build_tensor_info(graph: tf.Graph, info: NodeInfo) -> TensorInfo:
    if info is not None:
        tensor = graph.get_tensor_by_name(info.tensor)
        return TensorInfo(
            dtype=tf.dtypes.as_dtype(tensor.dtype).as_datatype_enum,
            tensor_shape=tf.TensorShape(tensor.shape).as_proto(),
            name=info.tensor)
    else:
        return None


def get_input_nodes(graph: Union[tf.Graph, GraphDef]) -> List[NodeInfo]:
    """
    Return information about a graph's inputs.

    Args:
        graph: Graph or GraphDef object

    Returns:
        List of NodeInfo tuples holding name, shape, and type of the input
    """
    if isinstance(graph, tf.Graph):
        graph_def = graph.as_graph_def()
    else:
        graph_def = graph

    def is_input(node): return node.op == c.TFJS_NODE_PLACEHOLDER_KEY
    return [_node_info(node) for node in graph_def.node if is_input(node)]


def get_output_nodes(graph: Union[tf.Graph, GraphDef]) -> List[NodeInfo]:
    """
    Return information about a graph's outputs.

    Args:
        graph: Graph or GraphDef object

    Returns:
        List of NodeInfo tuples holding name, shape, and type of the input;
        shape will be left empty
    """
    # normalise input
    if isinstance(graph, tf.Graph):
        graph_def = graph.as_graph_def()
    else:
        graph_def = graph
    # visit graph nodes and test for references
    # assumption: all referenced nodes are declared *before* use
    ops = _op_nodes(graph_def)
    outputs = []
    for i, node in enumerate(ops):
        has_ref = False
        for test in ops[i+1:]:
            if node.name in test.input:
                has_ref = True
                break
        if not has_ref:
            outputs.append(_node_info(node))
    return outputs


def get_input_tensors(graph: Union[tf.Graph, GraphDef]) -> List[str]:
    """
    Return the names of the graph's input tensors.

    Args:
        graph: Graph or GraphDef object

    Returns:
        List of tensor names
    """
    return [node.tensor for node in get_input_nodes(graph)]


def get_output_tensors(graph: Union[tf.Graph, GraphDef]) -> List[str]:
    """
    Return the names of the graph's output tensors.

    Args:
        graph: Graph or GraphDef object

    Returns:
        List of tensor names
    """
    return [node.tensor for node in get_output_nodes(graph)]


def infer_signature(graph: tf.Graph) -> Optional[SignatureDef]:
    """
    Return the graph's signature (inputs and outputs) as a TF SignatureDef.

    Args:
        graph: Graph object

    Returns:
        TF SignatureDef from Graph; None, iff signature couldn't be inferred
    """
    inputs = {info.name: _build_tensor_info(graph, info)
              for info in get_input_nodes(graph)}
    outputs = {info.name: _build_tensor_info(graph, info)
               for info in get_output_nodes(graph)}
    if all(tensor_info is not None for tensor_info in outputs.values()):
        # only valid if we could infer all output shapes
        return SignatureDef(inputs=inputs, outputs=outputs,
                            method_name=tf.saved_model.PREDICT_METHOD_NAME)
    else:
        return None


def rename_input_nodes(graph_def: GraphDef, name_mapping: dict) -> GraphDef:
    """Rename input nodes - the update is performed in-place.

        Args:
            graph_def: GraphDef proto containing the model
            name_mapping: dict that maps input names to new names

        Returns:
            Updated GraphDef proto - same as input since the update is
            performed in-place.
    """
    _validate_mapping(graph_def, get_input_nodes(graph_def), name_mapping)
    renamed_nodes = name_mapping.keys()
    for node in graph_def.node:
        if node.name in renamed_nodes:
            node.name = name_mapping[node.name]
        else:
            for idx, name in enumerate(node.input):
                if name in renamed_nodes:
                    node.input[idx] = name_mapping[name]
    return graph_def


def rename_output_nodes(graph_def: GraphDef, name_mapping: dict) -> GraphDef:
    """Rename output nodes - the update is performed in-place.

        Args:
            graph_def: GraphDef proto containing the model
            name_mapping: dict that maps output names to new names

        Returns:
            Updated GraphDef proto - same as input since the update is
            performed in-place.
    """
    output_nodes = get_output_nodes(graph_def)
    _validate_mapping(graph_def, output_nodes, name_mapping)
    nodes_to_rename = name_mapping.keys()
    target_nodes = filter(lambda n: n.name in nodes_to_rename, graph_def.node)
    for node in target_nodes:
        new_name = name_mapping[node.name]
        if node.op == 'Identity':
            # dummy nodes can just be renamed
            node.name = new_name
        else:
            # append an identity node that takes the original output as input
            identity = rewrite.make_op_node('Identity', node, name=new_name,
                                            dtype=_dtype(node))
            graph_def.node.append(identity)
    return graph_def


def _validate_mapping(graph_def: GraphDef, valid_nodes: List[NodeInfo],
                      mapping: dict):
    valid_names = set(info.name for info in valid_nodes)
    existing_nodes = set(node.name for node in graph_def.node)
    inverted_mapping = defaultdict(list)
    for old_name, new_name in mapping.items():
        if old_name not in valid_names:
            raise ValueError(f'"{old_name}" is not a valid node name. '
                             f'Valid names are {str(valid_names)[1:-1]}')
        if new_name in existing_nodes:
            raise ValueError(f'"{old_name}" cannot be renamed to "{new_name}":'
                             ' the name is already in use.')
        already_mapped = inverted_mapping[new_name]
        if len(already_mapped) > 0:
            mapped_to = already_mapped[0]
            raise ValueError(f'"{old_name}" cannot be renamed to "{new_name}":'
                             f' the name is already mapped to "{mapped_to}"')
        else:
            already_mapped.append(old_name)
