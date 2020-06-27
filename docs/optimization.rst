Module ``tfjs_graph_converter.optimization``
===============================================

The optimization module ``tfjs_graph_converter.optimization`` contains
graph-optimization functions. It is used to clean-up graphs that have
been rewritten to enable TFJS functions that are not supported in TF.

..

    The module is considered private to the converter so expect the
    interface to change between versions. Use it at your own risk.

.. contents:: **Table of Contents**
    :backlinks: none


``optimize_graph``
^^^^^^^^^^^^^^^^^^

.. code:: python

   optimize_graph(
        graph: tf.Graph,
        target: str = None
   ) -> GraphDef

Optimizes a TF frozen graph by running TF's integrated optimization
functionality (*"Grappler"*). The resulting ``GraphDef`` message must be
converted to a ``tf.Graph`` or TF2 function before it can be used for
inference. The function result can be serialized as-is to a file, however. 

..

    **Arguments:**

**graph**
    ``tf.Graph`` instance that holds a frozen graph model including all
    weights.

**level**
    Optimization target (*None*, *CPU*, or *GPU*). This parameter is reserved
    for future use and not currently supported.

..

    **Returns:**

The function returns the optimized graph as a ``GraphDef`` protobuf message
for serialization.

..

    **Example:**

.. code:: python

    import tensorflow as tf
    import tfjs_graph_converter as tfjs

    graph = tfjs.api.load_graph_model('./models/some_tfjs_graph_model/')
    # do some model surgery 
    updated_graph = extract_hidden_layers(graph)
    graph_def = tfjs.optimization.optimize(updated_graph)
    # serialize optimized graph to file
    tf.io.write_graph(graph_def, './models/updated_model/',
                      'frozen_graph.pb', as_text=False)
