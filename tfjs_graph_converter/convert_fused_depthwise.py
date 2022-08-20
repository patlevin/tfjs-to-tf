# SPDX-License-Identifier: MIT
# Copyright © 2022 Patrick Levin
"""Functions to rewrite FusedDepthwiseConv2d as native TensorFlow operations"""

import tfjs_graph_converter.graph_rewrite_util as util


def _split_fused_depthwise(node: util.NodeDef, *_) -> util.NodeList:
    """Decompose fused op into DepthwiseConv2dNative + BiasAdd [+ Activation]
    """
    fused_ops = list(s.decode('utf-8') for s in node.attr['fused_ops'].list.s)
    inputs = node.input

    op = 'DepthwiseConv2dNative'
    depthwise = util.make_op_node(op, inputs[0:2], f'{node.name}/conv2d')
    depthwise = util.copy_op_attrs(source=node, target=depthwise)
    op = fused_ops[0]
    bias_add = util.make_op_node(op, [depthwise, inputs[2]], f'{node.name}/{op}')
    bias_add = util.copy_op_attrs(source=node, target=bias_add)
    node_list = [depthwise, bias_add]
    if len(fused_ops) > 1:
        # we have an activation function
        op = fused_ops[1]
        input_nodes = [bias_add] + inputs[3:]
        if util.get_op_def(op) is None:
            # unsupported activation function - just copy type attribute
            dtype = depthwise.attr['T'].type
            activation = util.make_op_node(
                op, input_nodes, f'{node.name}/{op}', dtype)
        else:
            # supported activation function - copy applicable attributes
            activation = util.make_op_node(
                op, input_nodes, f'{node.name}/{op}')
            activation = util.copy_op_attrs(source=node, target=activation)
        node_list.append(activation)
    return node_list


def split_fused_depthwise(input_graph_def: util.GraphDef) -> util.GraphDef:
    """Decompose all fused depthwise conv2d operations into separate operations

    This function looks for fused depthwise operations and splits matching
    nodes into individual operations.

    Fused activation functions that aren't supported (e.g. 'Prelu') can be
    replaced afterwards in a separate processing step.

    Args:
        input_graph_def: TF graph_def proto to be processed

    Returns:
        Updated copy of the input graph with matching nodes replaced by
        individual operations
    """
    return util.replace_matching_nodes(input_graph_def,
                                       util.is_fused_depthwise,
                                       _split_fused_depthwise)
