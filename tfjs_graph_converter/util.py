# SPDX-License-Identifier: MIT
# Copyright © 2020 Patrick Levin
from __future__ import absolute_import

from collections import namedtuple

import numpy as np
import tensorflow as tf

import tfjs_graph_converter.common as c

_DTYPE_MAP = [
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

def _is_op_node(node):
    return node.op not in (c.TFJS_NODE_CONST_KEY, c.TFJS_NODE_PLACEHOLDER_KEY)

def _op_nodes(graph_def):
    return [node for node in graph_def.node if _is_op_node(node)]

def _map_type(type_id):
    if type_id < 0 or type_id > len(_DTYPE_MAP):
        raise ValueError(f'Unsupported data type: {type_id}')
    np_type = _DTYPE_MAP[type_id]
    return np_type

def _get_shape(node):
    shape = lambda attr: attr.shape.dim
    size = lambda dim: dim.size if dim.size > 0 else None
    return [size(dim) for dim in shape(node.attr[c.TFJS_ATTR_SHAPE_KEY])]

def _node_info(node):
    dtype = lambda n: _map_type(n.attr[c.TFJS_ATTR_DTYPE_KEY].type)
    return NodeInfo(name=node.name, shape=_get_shape(node), dtype=dtype(node),
            tensor=node.name + ':0')

def get_input_nodes(graph):
    """
    Return information about a graph's inputs.

    Arguments:
        graph: Graph or GraphDef object

    Returns:
        List of NodeInfo objects holding name, shape, and type of the input
    """
    if isinstance(graph, tf.Graph):
        graph_def = graph.as_graph_def()
    else:
        graph_def = graph
    nodes = [n for n in graph_def.node if n.op == c.TFJS_NODE_PLACEHOLDER_KEY]
    return [_node_info(node) for node in nodes]

def get_output_nodes(graph):
    """
    Return information about a graph's outputs.

    Arguments:
        graph: Graph or GraphDef object

    Returns:
        List of NodeInfo objects holding name, shape, and type of the input;
        shape will be left empty
    """
    if isinstance(graph, tf.Graph):
        graph_def = graph.as_graph_def()
    else:
        graph_def = graph

    ops = _op_nodes(graph_def)
    outputs = []
    for i in range(len(ops)):
        node = ops[i]
        has_ref = False
        for test in ops[i+1:]:
            if node.name in test.input:
                has_ref = True
                break
        if not has_ref:
            outputs.append(node)

    return [_node_info(node) for node in outputs]

def get_input_tensors(graph):
    """
    Return the names of the graph's input tensors.

    Arguments:
        graph: Graph or GraphDef object

    Returns:
        List of tensor names
    """
    return [node.tensor for node in get_input_nodes(graph)]

def get_output_tensors(graph):
    """
    Return the names of the graph's output tensors.

    Arguments:
        graph: Graph or GraphDef object

    Returns:
        List of tensor names
    """
    return [node.tensor for node in get_output_nodes(graph)]
