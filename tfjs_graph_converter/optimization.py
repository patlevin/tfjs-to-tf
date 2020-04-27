# SPDX-License-Identifier: MIT
# Copyright © 2020 Patrick Levin
"""Graph optimization functions"""

from typing import List

import tensorflow as tf

from tensorflow.compat.v1 import GraphKeys
from tensorflow.core.framework.graph_pb2 import GraphDef
from tensorflow.core.protobuf.config_pb2 import ConfigProto
from tensorflow.core.protobuf.meta_graph_pb2 import SignatureDef
from tensorflow.python.framework import dtypes
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.training.saver import export_meta_graph

from tfjs_graph_converter import common as c
from tfjs_graph_converter.graph_rewrite_util import replace_matching_nodes
from tfjs_graph_converter.util import get_input_nodes, get_output_nodes


def _build_signature_def(graph: tf.Graph,
                         input_nodes: list,
                         output_nodes: list) -> SignatureDef:
    """Build model signature (input- and output descriptions) for a graph"""
    signature_def = SignatureDef()

    def add_tensor(nodes, info):
        nodes[info.name].name = info.name
        if info.dtype is not None:
            dtype = dtypes.as_dtype(info.dtype)
            shape = tf.TensorShape(info.shape)
            nodes[info.name].dtype = dtype.as_datatype_enum
            nodes[info.name].tensor_shape.CopyFrom(shape.as_proto())

    for input_info in input_nodes:
        op = graph.get_operation_by_name(input_info.name)
        if op.type != c.TFJS_NODE_CONST_KEY:
            add_tensor(signature_def.inputs, input_info)
    for output_info in output_nodes:
        add_tensor(signature_def.outputs, output_info)
    return signature_def


def _to_node_name(tensor_name: str) -> str:
    """Remove port from tensor name to give node name"""
    return tensor_name.split(':')[0]


def _mark_outputs_as_train_op(graph: tf.Graph,
                              signature_def: SignatureDef) -> None:
    """Mark output nodes as training ops, so the optimizer ignores them"""
    train_op = GraphKeys.TRAIN_OP
    for _, tensor in signature_def.outputs.items():
        name = _to_node_name(tensor.name)
        graph.add_to_collection(train_op, graph.get_operation_by_name(name))


def _remove_unused_control_flow_inputs(graph_def: GraphKeys) -> GraphDef:
    """The graph optimizer marks unsused nodes, which we can remove
       from the graph
    """
    def is_unused(node):
        return (node.op == c.TFJS_NODE_PLACEHOLDER_KEY
                and node.name.startswith('unused_control_flow_input'))

    result, _ = replace_matching_nodes(graph_def, is_unused, lambda _: [])
    return result


def _run_tf_optimizer(config: ConfigProto,
                      graph: tf.Graph,
                      signature_def: SignatureDef) -> GraphDef:
    """Run the TF optimizer ("grappler") on a graph"""
    graph_def = graph.as_graph_def()
    meta_graph = export_meta_graph(graph_def=graph_def, graph=graph)
    meta_graph.signature_def['not_used_key'].CopyFrom(signature_def)
    return tf_optimizer.OptimizeGraph(config, meta_graph)


def _set_optimization_options(config: ConfigProto, options: List[str]) -> None:
    """Set options for the graph optimizer"""
    rewriter_config = config.graph_options.rewrite_options
    rewriter_config.optimizers[:] = options


def optimize_graph(graph: tf.Graph, level=None) -> GraphDef:
    """Optimise a tensorflow graph for inference after modification

    This function optimises the given graph for inference after the graph
    may have been modified to replace known, but unsupported operations.
    Optimisation might use multiple passes and aim at CPUs or GPUs.

    Args:
        graph: Tensorflow v1 graph (or wrapped v2 function) to be optimised
        level: optional optimisation level; currently unsupported

    Returns:
        Optimised ``GraphDef`` message for inference or format conversion
    """
    inputs = get_input_nodes(graph)
    outputs = get_output_nodes(graph)
    signature_def = _build_signature_def(graph, inputs, outputs)
    _mark_outputs_as_train_op(graph, signature_def)
    config = ConfigProto()
    _set_optimization_options(config, [
        'debug_stripper', 'remap', 'constfold', 'arithmetic', 'dependency'
    ])
    optimised_graph = _run_tf_optimizer(config, graph, signature_def)
    optimised_graph = _remove_unused_control_flow_inputs(optimised_graph)
    return optimised_graph
