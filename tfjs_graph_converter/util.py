# SPDX-License-Identifier: MIT
# Copyright © 2020 Patrick Levin
"""Utility functions for working with TensorFlow Graphs"""
from __future__ import absolute_import

from collections import namedtuple
from typing import Dict, List, Optional, Union

import numpy as np
import tensorflow as tf
from tensorflow.core.framework.graph_pb2 import GraphDef
from tensorflow.core.framework.node_def_pb2 import NodeDef
from tensorflow_core.core.protobuf.meta_graph_pb2 import SignatureDef
from tensorflow_core.core.protobuf.meta_graph_pb2 import TensorInfo

import tfjs_graph_converter.common as c

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
    return [size(dim) for dim in shape(node.attr[c.TFJS_ATTR_SHAPE_KEY])]


def _get_tensor_shape(node: NodeDef) -> List[int]:
    def shape(attr): return attr.tensor.tensor_shape.dim if attr else None
    def size(dim): return dim.size if dim.size > 0 else None
    return [size(dim) for dim in shape(node.attr[c.TFJS_ATTR_VALUE_KEY])]


def _node_info(node: NodeDef) -> NodeInfo:
    def dtype(n): return _map_type(n.attr[c.TFJS_ATTR_DTYPE_KEY].type)
    return NodeInfo(name=node.name, shape=_get_shape(node), dtype=dtype(node),
                    tensor=node.name + ':0')


def _tensor_info(node: NodeDef) -> NodeInfo:
    def dtype(n): return _map_type(n.attr[c.TFJS_ATTR_DTYPE_KEY].type)
    return NodeInfo(name=node.name, shape=_get_tensor_shape(node),
                    dtype=dtype(node), tensor=node.name + ':0')


def _get_node_map(graph: Union[tf.Graph, GraphDef]) -> Dict[str, NodeDef]:
    if isinstance(graph, tf.Graph):
        graph_def = graph.as_graph_def()
    else:
        graph_def = graph
    return {node.name: node for node in graph_def.node}


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

    def is_input(node):
        return node.op == c.TFJS_NODE_PLACEHOLDER_KEY

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
            outputs.append(node)
    return [_node_info(node) for node in outputs]


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


def infer_signature(graph: Union[tf.Graph, GraphDef]
                    ) -> Optional[SignatureDef]:
    """
    Return the graph's signature (inputs and outputs) as a TF SignatureDef.

    Args:
        graph: Graph object

    Returns:
        TF SignatureDef from Graph; None, iff signature couldn't be inferred
    """
    def _build_tensor_info(node_info: NodeInfo) -> TensorInfo:
        if node_info is not None:
            return TensorInfo(
                dtype=tf.dtypes.as_dtype(node_info.dtype).as_datatype_enum,
                tensor_shape=tf.TensorShape(node_info.shape).as_proto(),
                name=node_info.tensor)
        else:
            return None

    def _get_tensor_shape_info(nodes: Dict[str, NodeDef], info: NodeInfo
                               ) -> TensorInfo:
        # heuristic: take the LAST input (skip control inputs) while op!=Const
        #            and tensor shape not set
        name = info.name
        if nodes[name].op in ('Identity', 'Const') and len(info.shape) > 0:
            return info
        input_node = [n for n in reversed(nodes[name].input) if n[0] != '^'][0]
        return _get_tensor_shape_info(nodes, _tensor_info(nodes[input_node]))

    nodes = _get_node_map(graph)
    inputs = {info.tensor: _build_tensor_info(info)
              for info in get_input_nodes(graph)}
    # outputs are tricky - infer the shape from their inputs
    outputs = {info.tensor: _build_tensor_info(_get_tensor_shape_info(nodes,
                                               info))
               for info in get_output_nodes(graph)}
    if all(tensor_info is not None for tensor_info in outputs.values()):
        # only valid if we could infer all output shapes
        return SignatureDef(inputs=inputs, outputs=outputs,
                            method_name=tf.saved_model.PREDICT_METHOD_NAME)
    else:
        return None
