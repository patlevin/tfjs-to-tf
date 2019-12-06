# Copyright (c) 2019 Patrick Levin
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================
from __future__ import absolute_import

from collections import namedtuple

import numpy as np
import tensorflow as tf

import tfjs_graph_converter.common as common

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
    return node.op not in (common.TFJS_NODE_CONST_KEY, common.TFJS_NODE_PLACEHOLDER_KEY)

def _op_nodes(graph_def):
    return [node for node in graph_def.node if _is_op_node(node)]

def _map_type(type_id):
    if type_id < 0 or type_id > len(_DTYPE_MAP):
        raise ValueError("Unsupported data type: {}".format(type_id))
    np_type = _DTYPE_MAP[type_id]
    return np_type

def _get_shape(node):
    shape = lambda attr: attr.shape.dim
    size = lambda dim: dim.size if dim.size > 0 else None
    return [size(dim) for dim in shape(node.attr[common.TFJS_ATTR_SHAPE_KEY])]

def _node_info(node):
    dtype = lambda n: _map_type(n.attr[common.TFJS_ATTR_DTYPE_KEY].type)
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
    nodes = [n for n in graph_def.node if n.op in (common.TFJS_NODE_PLACEHOLDER_KEY)]
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
    for i in range(0, len(ops)):
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
