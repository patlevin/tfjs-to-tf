# SPDX-License-Identifier: MIT
# Copyright © 2020 Patrick Levin
"""Utility functions for TFJS compatibility"""

import re
from collections import defaultdict
from typing import Dict, Tuple
from tfjs_graph_converter import util
from tfjs_graph_converter import graph_rewrite_util as r
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import op_def_registry

_DT_INT32 = dtypes.int32.as_datatype_enum


def _get_tensor_name(node: r.NodeDef, channel: int) -> str:
    return node.name if channel == 0 else f'{node.name}:{channel}'


def _set_tensor_dtype(tensor_node: r.NodeDef, dtype: int) -> None:
    """Set the data type of a tensor node (Const or Placeholder)"""
    tensor_node.attr['dtype'].type = dtype
    if 'value' in tensor_node.attr:
        tensor_node.attr['value'].tensor.dtype = dtype


def _make_cast_node(node: r.NodeDef, output_idx: int,
                    src_type: int, dst_type: int) -> r.NodeDef:
    """Return a cast operation for a given node"""
    input_name = _get_tensor_name(node, output_idx)
    suffix = f'To{dtypes.as_dtype(dst_type).name.capitalize()}'
    cast_op = r.NodeDef(op='Cast', name=f'{node.name}/{output_idx}/{suffix}')
    cast_op.input.extend([input_name])
    cast_op.attr['SrcT'].type = src_type
    cast_op.attr['DstT'].type = dst_type
    return cast_op


def _get_input_node(ref: r.NodeDef, idx: int, node_map: r.NameToNode
                    ) -> Tuple[r.NodeDef, int]:
    """Return node and ouptut index of node input idx"""
    m = re.match(r'([^:]+)(:(\d+))?', ref.input[idx])
    assert(m is not None)
    node = node_map[m.groups()[0]]
    ouput_idx = int(m.groups()[-1] or 0)
    return node, ouput_idx


def _get_output_type(node: r.NodeDef, output_idx: int) -> int:
    """Return the type of the nth output of a node"""
    op = op_def_registry.get(node.op)
    output_arg = op.output_arg[output_idx]
    if output_arg.type != 0:
        return output_arg.type
    elif len(output_arg.type_attr) > 0:
        return node.attr[output_arg.type_attr].type
    else:
        raise ValueError(f'cannot determine output type of node "{node.name}"'
                         f' op={op.name}')


def convert_int64_to_int32(graph_def: r.GraphDef) -> r.GraphDef:
    """Convert int64 input to int32 for TFJS compatibility

    Args:
        graph_def: GraphDef proto containing the network layout
    Returns:
        Updated graph with int64 inputs converted to int32
    """
    inputs = util.get_input_nodes(graph_def)
    convert = [info.name for info in inputs if info.dtype == util.np.int64]
    if len(convert) == 0:
        return graph_def
    # quick access to nodes by name
    node_map = r.get_input_node_map(graph_def)
    # map of all node inputs to their referencing node and their argument index
    input_map = defaultdict(list)
    for node in graph_def.node:
        for index, name in enumerate(node.input):
            input_map[name].append((index, node))
    # type cast ops to add to the graph
    type_cast_ops = []
    # nodes that require a type cast op
    type_cast_candidates: Dict[str, Tuple[int, r.NodeDef]] = {}

    for node in map(lambda x: node_map[x], convert):
        _set_tensor_dtype(node, _DT_INT32)
        # find all nodes that reference this input and adjust their datatype
        # attributes if required
        # technical note: referenced_by is a stack, this really is a
        # depth-first recursion
        referenced_by = input_map[node.name]
        while len(referenced_by) > 0:
            idx, ref = referenced_by.pop()
            # get the input node and the index of the output tensor
            input_node, output_idx = _get_input_node(ref, idx, node_map)
            # find the description of this node's operation
            op = op_def_registry.get(ref.op)
            desc = op.input_arg[idx]
            # find out whether we can just change the input type and which
            # attributes we might need to touch
            if desc.type != 0 and desc.type != _DT_INT32:
                # input type is fixed and cannot be changed: add a type cast
                cast_op = _make_cast_node(input_node, output_idx, _DT_INT32,
                                          desc.type)
                ref.input[idx] = cast_op.name
                type_cast_ops.append(cast_op)
                node_map[cast_op.name] = cast_op
                input_map[cast_op.name].append((idx, ref))
            elif desc.type_list_attr != '' or desc.type_attr == '':
                # input arrays of potentially mixed types cannot be handled
                raise ValueError("don't know how to handle input type changes"
                                 f' for node "{ref.name}" op={ref.op}')
            else:
                # change the type of this input
                type_attr = desc.type_attr
                ref.attr[type_attr].type = _DT_INT32
                if ref.name in type_cast_candidates:
                    del type_cast_candidates[ref.name]
                # check the other inputs for type compatibility
                for i, desc in enumerate(op.input_arg):
                    if i == idx or desc.type_attr != type_attr:
                        continue    # not a matching input
                    input_node, output_idx = _get_input_node(ref, i, node_map)
                    if input_node.name in convert:
                        continue    # Placeholder that will be converted
                    src_type = _get_output_type(input_node, output_idx)
                    if src_type == _DT_INT32:
                        continue    # type matches already
                    if input_node.op == 'Const':
                        # weight tensor: harmonize_dtypes() will fix these
                        _set_tensor_dtype(input_node, _DT_INT32)
                    else:
                        # add node as a candidate for needing type cast op
                        type_cast_candidates[input_node.name] = (i, ref)
                # process any changed outputs next
                for idx, output in enumerate(op.output_arg):
                    if output.type_attr == type_attr:
                        input_name = _get_tensor_name(ref, idx)
                        referenced_by += input_map[input_name]

    for idx, ref in type_cast_candidates.values():
        # add type cast operations for all nodes that have a type mismatch
        inp_node, channel = _get_input_node(ref, idx, node_map)
        src_type = _get_output_type(inp_node, channel)
        if src_type != _DT_INT32:
            cast_op = _make_cast_node(inp_node, channel, src_type, _DT_INT32)
            ref.input[idx] = cast_op.name
            type_cast_ops.append(cast_op)
            node_map[cast_op.name] = cast_op

    graph_def.node.extend(type_cast_ops)
    return graph_def
