Module ``tfjs_graph_converter.util``
====================================

Utility functions can be found in the module ``tfjs_graph_converter.util``.
The functions in this module are used internally, but can be useful for
working with imported models as well.

.. contents:: **Table of Contents**
    :backlinks: none


``Types``
^^^^^^^^^

.. _NodeInfo:

============= ===========================================================
Type          Description
============= ===========================================================
**NodeInfo**  ``namedtuple`` for describing input and output nodes in
              TF v1 frozen graphs. Members are:

              ``name``
                Name of the graph node
              ``shape``
                List containing the node's tensor dimensions. Unspecified
                dimensions (e.g. batch sizes) are denoted by ``None``
              ``dtype``
                TF DType enum that specifies the type of the node's
                tensor elements
              ``tensor``
                The name of the tensor associated with the node 
============= ===========================================================

``get_input_nodes``
^^^^^^^^^^^^^^^^^^^

.. code:: python

   get_input_nodes(
        graph: Union[tf.Graph, GraphDef]
   ) -> List[NodeInfo]

Analyzes the given frozen graph or ``GraphDef`` message and returns information
about all input nodes.

..

    **Arguments:**

**graph**
    Frozen graph or ``GraphDef`` message.

..

    **Returns:**

List of NodeInfo_ tuples for each model input.

..

    **Example:**

.. code:: python

    import tfjs_graph_converter.api as tfjs
    import tfjs_graph_converter.util as tfjs_util

    MODEL_PATH = '~/models/tfjs_model/'

    # load tfjs graph model
    graph = tfjs.load_graph_model(MODEL_PATH)
    model_inputs = tfjs_util.get_input_nodes(graph)
    model_outputs = tfjs_util.get_output_nodes(graph)

    print(f'Model inputs:{model_inputs}')
    print'f'Model outputs: {model_outputs}')

``get_output_nodes``
^^^^^^^^^^^^^^^^^^^

.. code:: python

   get_output_nodes(
        graph: Union[tf.Graph, GraphDef]
   ) -> List[NodeInfo]

Analyzes the given frozen graph or ``GraphDef`` message and returns information
about all output nodes.

..

    **Arguments:**

**graph**
    Frozen graph or ``GraphDef`` message.

..

    **Returns:**

List of NodeInfo_ tuples for each model output. Note that the ``shape``
value will be empty so you cannot use this function to infer the output shapes
of a frozen graph model. 

..

    **Example:**

See `get_input_nodes`_ for an example.


``get_input_tensors``
^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   get_input_tensors(
        graph: Union[tf.Graph, GraphDef]
   ) -> List[str]

Analyzes the given frozen graph or ``GraphDef`` message and returns the names
of all input tensors.

..

    **Arguments:**

**graph**
    Frozen graph or ``GraphDef`` message.

..

    **Returns:**

List of tensor names for each model input for use with TF v1 inference using
the ``feed_dict`` parameter.

..

    **Example:**

.. code:: python

    import tensorflow as tf
    import tfjs_graph_converter.api as tfjs
    import tfjs_graph_converter.util as tfjs_util

    MODEL_PATH = '~/models/tfjs_model/'

    # load tfjs graph model and get the tensor names
    graph = tfjs.load_graph_model(MODEL_PATH)
    input_names = tfjs_util.get_input_tensors(graph)
    output_names = tfjs_util.get_output_tensors(graph)

    data = load_data(...)

    input_tensor = input_names[0]  # e.g. single input
    with tf.compat.v1.Session(graph=graph):
        results = sess.run(output_names, feed_dict={input_tensor: data})

    show_results(results)

``get_output_tensors``
^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   get_output_tensors(
        graph: Union[tf.Graph, GraphDef]
   ) -> List[str]

Analyzes the given frozen graph or ``GraphDef`` message and returns the tensor
names of all model outputs.

..

    **Arguments:**

**graph**
    Frozen graph or ``GraphDef`` message.

..

    **Returns:**

List of tensor names for each model output. This function can be used to
determine the names of requested model outputs.

..

    **Example:**

See `get_input_tensors`_ for an example.


``infer_signature``
^^^^^^^^^^^^^^^^^^^^

.. code:: python

   infer_signature(
        graph: Union[tf.Graph, GraphDef]
   ) -> Optional[SignatureDef]

Analyzes the given frozen graph or ``GraphDef`` message and returns the
``SignatureDef`` for use with TF ``SavedModel``.

..

    **Arguments:**

**graph**
    Frozen graph or ``GraphDef`` message.

..

    **Returns:**

``SignatureDef`` containing the inputs and outputs of the model. The method
name is fixed to the TF default prediction model signature name. ``None`` is
returned, if no output tensor shape could be determined.

..

    **Example:**

.. code:: python

    import tensorflow as tf
    import tfjs_graph_converter.api as tfjs
    import tfjs_graph_converter.util as tfjs_util

    MODEL_PATH = '~/models/tfjs_model/'

    # load tfjs graph model and get the signature
    graph = tfjs.load_graph_model(MODEL_PATH)
    signature_def = tfjs_util.infer_signature(graph)
    # change the signature name, e.g. for use with saved_model 
    signature_def.method_name = 'my_model/predict'
