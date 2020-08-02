# SPDX-License-Identifier: MIT
# Copyright © 2020 Patrick Levin
"""Utility functions for rewriting TensorFow Graphs"""

from typing import Any, Callable, Dict, List, Optional, Text, Tuple, Union

from tensorflow import as_dtype
from tensorflow.core.framework.attr_value_pb2 import AttrValue
from tensorflow.core.framework.graph_pb2 import GraphDef
from tensorflow.core.framework.node_def_pb2 import NodeDef
from tensorflow.core.framework.op_def_pb2 import OpDef
from tensorflow.core.framework.tensor_pb2 import TensorProto
from tensorflow.core.framework.tensor_shape_pb2 import TensorShapeProto

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import op_def_registry

from numpy import ndarray as Tensor

# common type definitions
NodeList = List[NodeDef]
NameOrNode = Union[Text, NodeDef]
NameToNode = Dict[Text, NodeDef]
InputList = List[NameOrNode]
Inputs = Union[NameOrNode, InputList]
WeightTransform = Callable[[Tensor], Tensor]
WeightModifiers = Dict[Text, WeightTransform]
NodeTransform = Callable[[NodeDef, NameToNode, WeightModifiers], NodeList]


def get_op_def(op_name: Text) -> Optional[OpDef]:
    """
    Get the definition for a native TF operation.
    This is useful for checking whether an operation is supported or
    to get all valid inputs and attributes.

    Args:
        op_name: Name of the native TF operation (e.g. "AddV2")

    Returns:
        Protobuf object containing the operation definition
        `None` is returned, if the operation is not registered with TF
    """
    return op_def_registry.get(op_name)


def make_op_node(op_name: Text, inputs: Inputs, name: Text = None,
                 dtype: Any = None) -> NodeDef:
    """
    Create a TF graph node given the operation, input, and a name.
    The resulting node definition won't include any operation-specific
    attributes. It returns a valid node for most operations, though.

    Args:
        op_name: Native TF operation name (e.g. "MatMul")
        inputs: Input node, node name, or list of inputs nodes or node names
        name: Node name in the graph, must be unique and defaults to the
              operation name
        dtype: Optional data type of the operation (default: float32)

    Returns:
        TF graph node definition for the given operation, inputs, and name
    """
    input_list = inputs
    # convert scalar input into list
    if not isinstance(inputs, list):
        input_list = [input_list]
    # convert list items to strings
    for i, item in enumerate(input_list):
        if hasattr(item, 'name'):
            input_list[i] = item.name
    # generate node defintion
    if dtype is None:
        dtype = dtypes.float32.as_datatype_enum
    elif hasattr(dtype, 'as_datatype_enum'):
        dtype = dtype.as_datatype_enum
    else:
        dtype = dtypes.as_dtype(dtype).as_datatype_enum

    node_def = NodeDef(op=op_name, name=name or op_name,
                       attr={'T': AttrValue(type=dtype)})
    node_def.input.extend(input_list)
    return node_def


def make_const_node(data: Tensor, name: str = None) -> NodeDef:
    """
    Create a TF graph node containing a constant value.
    The resulting node is equivalent to using `tf.constant` on the
    default graph.

    Args:
        data: Numpy-array containing the data, shape, and datatype
        name: Optional name of the node

    Returns:
        Graph node for adding to a TF Graph instance
    """
    dtype = as_dtype(data.dtype).as_datatype_enum
    tensor_content = data.tobytes()
    tensor_dim = [TensorShapeProto.Dim(size=size) for size in data.shape]
    tensor_shape = TensorShapeProto(dim=tensor_dim)
    tensor_proto = TensorProto(tensor_content=tensor_content,
                               tensor_shape=tensor_shape,
                               dtype=dtype)
    node_def = NodeDef(op='Const', name=name or 'Const',
                       attr={
                           'value': AttrValue(tensor=tensor_proto),
                           'dtype': AttrValue(type=dtype)
                        })
    return node_def


def copy_op_attrs(source: NodeDef = None, target: NodeDef = None) -> NodeDef:
    """
    Copy valid node attributes from one node to another.
    Only attributes supported by the target node's operation will be copied.
    This is useful when splitting fused operations to retain the attributes
    of the original, non-fused operation in the isolated target node.

    Existing attributes will be overridden.

    Args:
        source: Graph node containing attributes to copy
        target: Graph node to copy the attributes to

    Returns:
        The updated target node.
    """
    op_def = get_op_def(target.op)
    if op_def is None:
        raise ValueError(f'Node {target.name}: unknown op name {target.op}')
    attrs_to_copy = set(attr.name for attr in op_def.attr)
    for key in source.attr:
        if key in attrs_to_copy:
            target.attr[key].CopyFrom(source.attr[key])
    return target


def update_graph_def(input_graph_def: GraphDef,
                     nodes_to_remap: Dict[Text, List[NodeDef]],
                     inputs_to_replace: Dict[Text, Text]) -> GraphDef:
    """
    Update a TF graph_def by replacing nodes and node inputs.
    There will be no consistency check in this function.
    Callers have to make sure the given remappings and input replacements
    result in a valid graph.

    Args:
        input_graph_def: TF graph_def with nodes or node inputs to replace
        nodes_to_remap: `dict` that maps node names to a list of replacement
            nodes. Nodes whose name map to an empty list, will be
            removed from the returned graph.
            Nodes that are not in the input graph_def but have an
            entry in the remap dict, will be ignored.
        inputs_to_replace: `dict` that maps node names to replacement names.
            Nodes that have been removed need to be replaced in all referenced
            graph nodes. This mapping can be used to make sure this happens.

    Returns:
        An updated copy of the input graph_def. The original inputs remains
        unchanged.
    """
    result_graph_def = GraphDef()
    for node in input_graph_def.node:
        if node.name in nodes_to_remap:
            nodes_to_insert = nodes_to_remap[node.name]
            if nodes_to_insert and len(nodes_to_insert) > 0:
                result_graph_def.node.extend(nodes_to_insert)
            continue
        new_node = NodeDef()
        new_node.CopyFrom(node)
        for i, input_node in enumerate(new_node.input):
            if input_node in inputs_to_replace:
                new_node.input[i] = inputs_to_replace[input_node]
        result_graph_def.node.extend([new_node])
    result_graph_def.versions.CopyFrom(input_graph_def.versions)
    return result_graph_def


def get_input_node_map(input_graph_def: GraphDef) -> NameToNode:
    """
    Return a mapping from node names to node_def instances from a given
    graph_def.
    The result can be used to check whether a node name is referenced in
    the graph or to quickly lookup the node_def given a node name.

    Args:
        input_graph_def: TF graph_def containing the nodes to generate the
                         mapping from.

    Returns:
        `dict` that maps node names to the corresponding node_def instances.
    """
    input_node_map = dict()
    for node in input_graph_def.node:
        if node.name not in input_node_map:
            input_node_map[node.name] = node
        else:
            raise ValueError(f'Duplicate node name: {node.name}')
    return input_node_map


def replace_matching_nodes(input_graph_def: GraphDef,
                           predicate: Callable[[NodeDef], bool],
                           transform: NodeTransform
                           ) -> Tuple[GraphDef, WeightModifiers]:
    """
    Replace all nodes that match a given predicate using the provided
    transformation function and return the new graph.
    The transformation function can also register a function to modify
    existing node weights or variables.

    Args:
        input_graph_def: TF graph_def to traverse and possibly modify
        predicate: a callable that takes a node_def and returns `True`
            if the node should be transformed
        transform: a callable that receives a graph node, a node map
            containing the names of all current graph nodes, and a
            `dict` that can be used to add node weight modifiers.
            The function is expected to return a list of nodes that
            replace the given node.
            The first node of this list is expected to receive at least
            one input of the replaced node.
            The last node in the returned list is expected to replace all
            references to the original node.

    Returns:
        Updated copy of the input graph with matching nodes replaced by
        the output of the transform function.
    """
    input_node_map = get_input_node_map(input_graph_def)
    nodes_to_remap = {}
    inputs_to_remap = {}
    weight_modifiers = {}
    for node in input_graph_def.node:
        if predicate(node):
            # the name of a replaced node becomes available for use right away
            del input_node_map[node.name]
            new_nodes = transform(node, input_node_map, weight_modifiers)
            nodes_to_remap[node.name] = new_nodes
            if new_nodes and len(new_nodes) > 0:
                # by convention, the output of the last node in the returned
                # sub-graph replaces the output of the original node
                inputs_to_remap[node.name] = new_nodes[-1].name
                # we need to update the input node map to avoid duplicate names
                for new_node in new_nodes:
                    input_node_map[new_node.name] = new_node
    output_graph_def = update_graph_def(input_graph_def, nodes_to_remap,
                                        inputs_to_remap)
    return output_graph_def, weight_modifiers


def generate_name_from(base_name: Text,
                       input_node_map: NameToNode,
                       suffix: Optional[Text] = None) -> Text:
    """
    The tfjs converter generates names separated by forward slashes ('/').
    We can get the root name of an operation node by splitting off the last
    part (optionally with an added suffix):

    prefix/model_name/node_name/layer_name -> prefix/model_name/node_name

    The resulting name is then made unique with repespect to the given node
    mapping.

    Args:
        base_name: Node name to generate the new name from
        input_node_map: `dict` containing the names of all nodes in the graph
        suffix: optional suffix that will replace the removed part of the
                original name

    Returns:
        Root name of the given node name; guaranteed to be unique within the
        graph.
    """
    base_name = '/'.join(base_name.split('/')[:-1]) or base_name
    if suffix:
        base_name = base_name + '/' + suffix
    target_name = base_name
    count = 0
    while target_name in input_node_map:
        count += 1
        target_name = f'{base_name}_{count}'
    return target_name


def is_fused_op(node: NodeDef, op_name: Text, activation: Text) -> bool:
    """
    Return whether a node represents a fused TF operation.

    Args:
        node: Node defintion
        op_name: Fused operation name (e.g. 'MatMul')
        activation: Name of the fused activation function (e.g. 'Relu')

    Returns:
        `True`, iff the node is a fused operation with the given activation
    """
    if node.op == f'_Fused{op_name}' and 'fused_ops' in node.attr:
        fused_ops = node.attr['fused_ops'].list.s
        return (len(fused_ops) == 2
                and fused_ops[0] in (b'BiasAdd', b'BiasAddV1')
                and fused_ops[1] == activation)
    return False


def is_fused_conv2d(node: NodeDef, activation: Text) -> bool:
    """Return whether a node is a fused conv2d operation with given activation
    """
    return is_fused_op(node, 'Conv2D', activation)


def is_fused_matmul(node: NodeDef, activation: Text) -> bool:
    """Return whether a node is a fused matmul operation with given activation
    """
    return is_fused_op(node, 'MatMul', activation)


def validate_supported_ops(input_graph_def: GraphDef) -> None:
    """
    Iterate through all graph nodes and validate operation names.

    Args:
        input_graph_def: Input graph to validate

    Raises:
        ValueError: the graph contains an unsupported operation
    """
    for node in input_graph_def.node:
        if not get_op_def(node.op):
            raise ValueError(f'Node {node.name}: unsupported op {node.op}')
        if 'fused_ops' in node.attr:
            # check all fused operations as well
            fused_ops = list(node.attr['fused_ops'].list.s)
            unsupported_ops = [op for op in fused_ops if not get_op_def(op)]
            if any(unsupported_ops):
                raise ValueError(f'Node {node.name}: unsupported fused op '
                                 f'{unsupported_ops[0]}')
