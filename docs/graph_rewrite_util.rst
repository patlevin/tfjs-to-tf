Module ``tfjs_graph_converter.graph_rewrite_util``
==================================================

This module contains functions for rewriting protobuf messages of TF graphs
directly. The functions operate on serialized graphs and don't rely on any
TF functionality.

..

    The module is considered private to the converter so expect the
    interface to change between version. Use it at your own risk.

.. contents:: **Table of Contents**
    :backlinks: none

Types
^^^^^

==================== ==========================================================
Type                 Description
==================== ==========================================================
**NodeList**         Alias for ``List[NodeDef]`` - a list of graph nodes
-------------------- ----------------------------------------------------------
**NameOrNode**       Alias for ``Union[str, NodeDef]`` - graph node or its name
-------------------- ----------------------------------------------------------
**NameToNode**       Alias for ``Dict[str, NodeDef]`` - map from node name
                     to node
-------------------- ----------------------------------------------------------
**InputList**        Alias for ``List[NameOrNode]`` - a list of graph nodes or
                     node names
-------------------- ----------------------------------------------------------
**Inputs**           Alias for ``Union[NameOrNode, InputList]`` - a type that
                     is used for function parameters that are lists of nodes.
                     These nodes can be represented by actual graph nodes, node
                     names or just a single value (graph node or node name).
-------------------- ----------------------------------------------------------
**WeightTransform**  Alias for ``Callable[[Tensor], Tensor]`` - a function that
                     transforms models weights. The function takes a ``numpy``
                     -array and returns a modified ``numpy``-array (or ``None``
                     to signal that a weight can be removed)
-------------------- ----------------------------------------------------------
**WeightModifiers**  Alias for ``Dict[str, WeightTransform]`` - maps weight
                     names to transform functions. Functions that receive a
                     parameter of thsi type can add items to store processing
                     steps.
-------------------- ----------------------------------------------------------
**NodeTransform**    Alias for
                     ``Callable[[NodeDef, NameToNode, WeightModifiers]],``
                     `` NodeList]``
                     - signature of a function for transforming nodes.
                     The function receives a graph node, a map of all nodes in
                     the graph, and a map for storing model weight modifiers.
                     The function returns a list of graph nodes to replace the
                     given node with.
-------------------- ----------------------------------------------------------
**Tensor**           Alias for ``numpy.ndarray``
-------------------- ----------------------------------------------------------
**TensorDict**       Alias for ``Dict[str, tensorflow.Tensor]``
==================== ==========================================================

``get_op_def``
^^^^^^^^^^^^^^^

.. code:: python

    get_op_def(
        op_name: str
    ) -> Optional[OpDef]

Gets the definition for a native TF operation. Useful to check whether an
operation is supported by TF or to query required attributes.

``make_op_node``
^^^^^^^^^^^^^^^^^

.. code:: python

    make_op_node(
        op_name: str,
        inputs: Inputs,
        name: str,
        dtype: Any
    ) -> NodeDef

Creates a TF graph node given the operation, inputs, and a name.
Inputs can be given as lists of node name (``str``), graph nodes (``NodeDef``),
or just a node name or node if only one input is required.

Name and dtype, are optional. ``dtype`` denotes the data type of the node and
can be specified as a ``DType``-instance, an enum value (integer), or a string.

``make_const_node``
^^^^^^^^^^^^^^^^^^^

.. code:: python

    make_const_node(
        data: Tensor,
        name: str
    ) -> NodeDef

Creates a TF graph node containing a constant value.

``copy_op_attrs``
^^^^^^^^^^^^^^^^^

.. code:: python

    copy_op_attrs(
        source: NodeDef,
        target: NodeDef
    ) -> NodeDef

Copies valid node attributes from one node to another. Used when separating
fused operations to copy attributes from the fused op to the separated op.

``update_graph_def``
^^^^^^^^^^^^^^^^^^^^

.. code:: python

    update_graph_def(
        input_graph_def: GraphDef,
        nodes_to_remap: Dict[str, NodeList],
        inputs_to_replace: Dict[str, str]
    ) -> GraphDef

Updates a TF frozen graph by replacing nodes and node inputs.
Nodes whose names match a key from ``nodes_to_remap`` are replaced by the mapped
list of nodes. The inputs of all graph nodes are tested against
``inputs_to_replace``. Matching input nodes are replaced by the mapped value
given in that parameter.

This does **not** apply to nodes in ``nodes_to_remap``, though! All nodes that
are values of ``nodes_to_remap`` are assumed to already have the correct
inputs wired into them.

``get_input_node_map``
^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    get_input_node_map(
        (input_graph_def: GraphDef
    ) -> NameToNode

Returns a mapping from node names to graph node instances from a given graph.
Checks whether node names are unique and raises a ``ValueError`` if duplicate
node names are found.

``replace_matching_nodes``
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    replace_matching_nodes(
        input_graph_def: GraphDef,
        predicate: Callable[[NodeDef], bool],
        transform: NodeTransform
    ) -> Tuple[GraphDef, WeightModifiers]

Replaces all nodes that match a given predicate using the provided
transformation function and return the new graph (and optionally
model weight modifiers).

``generate_name_from``
^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    generate_name_from(
        base_name: str,
        input_node_map: NameToNode,
        suffix: Optional[str]
    ) -> str

Utility to generate node names from nodes generated by TFJS from Keras
models. Returns unique node names given a map of nodes currently in the graph.

The function splits the ``base_name`` like ``os.path-split`` does and appends
``suffix`` if provided; ``model/layer/name`` becomes ``model/layer[/suffix]``.
If the resulting name is present in ``input_node_map``, a counter is appened
to it so that the returned name is unique with respect to ``input_node_map``.

``is_fused_op``
^^^^^^^^^^^^^^^

.. code:: python

    is_fused_op(
        node: NodeDef,
        op_name: str,
        activation: str
    ) -> bool

Returns whether a node is a fused operation with a given activation.
Allows for easy checking whether a graph contains a node with a fused
unsupported activation function that can be rewritten.

``validate_supported_ops``
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    validate_supported_ops(
        input_graph_def: GraphDef
    ) -> None

Iterates through all graph nodes and checks whether the node's operation is
actually supported by TF. Raises a ``ValueError`` if an unsupported operation
is found.

``harmonize_dtypes``
^^^^^^^^^^^^^^^^^^^^

.. code:: python

    harmonize_dtypes(
        graph_def: GraphDef,
        weights: Dict[str, tf.Tensor]
    ) -> Dict[str, tf.Tensor]

Iterates through a given weight dictionary and ensures that the type of weight
tensor elements and graph node attributes match. The graph remains unchanged,
while the tensor data is widened (or shortened) to match the graph node type.
The returned dictionary maps tensor names to tensor data that is guaranteed to
match the type of the corresponding graph node.
