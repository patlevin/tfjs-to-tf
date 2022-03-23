Module ``tfjs_graph_converter.api``
===================================

The public API can be found in the module ``tfjs_graph_converter.api``.
The functions in the API can be used to either add TFJS model support
directly to your Tensorflow programs, or to create your own converters.

.. contents:: **Table of Contents**
    :backlinks: none

``Types``
^^^^^^^^^

============== ==========================================================
Type           Description
============== ==========================================================
**GraphDef**   This type is the Tensorflow protobuf message for frozen
               graphs. It is used as input for functions like
               `graph_def_to_graph_v1`_.
-------------- ----------------------------------------------------------
**Tensor**     Alias for ``numpy.ndarray``.
-------------- ----------------------------------------------------------
**RenameMap**  Class for mapping model inputs and -outputs to new names.
               Pass string dictionaries to the constructor and use its
               ``apply()``-method to rename node in a model's ``GraphDef``
               proto.
-------------- ----------------------------------------------------------
**CompatMode** Enum that describes the compatibility mode for model
               conversion:

               NONE: use the full set of TF operations and optimisations

               TFJS: use harmonised data types for compatibility with
               older TFJS versions

               TFLITE: only use TFLite compatible ops
============== ==========================================================

``enable_cuda``
^^^^^^^^^^^^^^^^

.. code:: python

  enable_cuda() -> None

Enables CUDA support for the program.

..

    **Arguments:**

..

    **Returns:**

..

    **Details:**

By default, including the library will disable CUDA devices from being used.
This is done to enable the converter to process models that would otherwise
cause problems with certain CUDA-enabled devices due to lack of compute
capabilities or GPU memory constraints.

Since the library uses low-level Tensorflow APIs where device usage cannot 
be controlled from Python code, CUDA devices are disabled for the entire
process (script).

If your program requires CUDA support and includes this library, you can call
this function **before** any call to either Tensorflow or tfjs_graph_converter
functions.

..

    **Example:**

.. code:: python

    import sys
    from typing import List
    import tfjs_graph_converter.api as tfjs


    def main(args: List[str]) -> None:
        if '--enable-cuda' in args:
            # enable use of CUDA-capable devices
            tfjs.enable_cuda()
        graph = tfjs.load_graph_model('model-dir')
        model = tfjs.graph_def_to_function_v2(graph)
        inputs = ...
        # inference runs on CUDA-device if available and enabled
        result = model(inputs)
        # tf functions and libraries will be able to use GPUs if enabled


    if __name__ == '__main__':
        main(sys.argv)


``load_graph_model``
^^^^^^^^^^^^^^^^^^^^

.. code:: python

   load_graph_model(
        model_dir: str
   ) -> tf.Graph

Loads a tensorflowjs graph model from a directory and returns a TF v1
`tf.Graph`__ that can be used for inference.

..

    **Arguments:**

**model_dir**
    The directory that contains the ``model.json`` file.
    Alternatively, the path and name of the JSON file can be
    specified directly. Weight files must be located in the
    same directory as the model file.

..

    **Returns:**

``tf.Graph`` that contains the frozen graph and all model weights.

__ https://www.tensorflow.org/api_docs/python/tf/Graph

..

    **Example:**

.. code:: python

    import numpy as np
    import tensorflow as tf
    import tfjs_graph_converter.api as tfjs
    import tfjs_graph_converter.util as tfjs_util

    MODEL_PATH = '~/models/tfjs_graph_models/stylize_js/'
    SAMPLE_IMAGE = '~/images/samples/city.jpg'
    OUTPUT_IMAGE = '~/images/stylised/city-stylised.jpg'

    # load input image into numpy array
    img = tf.keras.preprocessing.image.load_img(SAMPLE_IMAGE)
    x = tf.keras.preprocessing.image.img_to_array(img, dtype=np.float32)
    # scale input to [0..1] and expand dimensions to 4d array (TF uses NHWC)
    x /= 255.0
    content_image = x[tf.newaxis, ...]

    # load tfjs graph model directly
    graph = tfjs.load_graph_model(MODEL_PATH)

    # evaluate the loaded model
    with tf.compat.v1.Session(graph=graph) as sess:
        # the module provides some helpers for querying model properties
        input_tensor_names = tfjs_util.get_input_tensors(graph)
        output_tensor_names = tfjs_util.get_output_tensors(graph)
        input_tensor = graph.get_tensor_by_name(input_tensor_names[0])

        results = sess.run(output_tensor_names,
                           feed_dict={input_tensor: content_image})
    # save the result
    stylised = np.squeeze(results)
    tf.keras.preprocessing.image.save_img(OUTPUT_IMAGE, stylised)


``load_graph_model_and_signature``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   load_graph_model_and_signature(
        model_dir: str,
        compat_mode: CompatMode = CompatMode.NONE
   ) -> Tuple[tf.Graph, Optional[SignatureDef]]

Loads a tensorflowjs graph model from a directory and returns a TF v1
`tf.Graph`__ that can be used for inference along with a TF `SignatureDef`__
that contains the inputs and outputs of the model.

..

    **Arguments:**

**model_dir**
    The directory that contains the ``model.json`` file.
    Alternatively, the path and name of the JSON file can be
    specified directly. Weight files must be located in the
    same directory as the model file.

**compat_mode**
    Set this argument to the requested mode to ensure that the resulting graph
    is compatible with TensorflowJS or TFLite if possible.

    CompatMode.NONE: the resulting model uses the full set of TF ops

    CompatMode.TFJS: the resulting model uses TFJS-compatible data types

    CompatMode.TFJS: the resulting model uses TFLite builtin ops only

..

    **Returns:**

``tf.Graph`` that contains the frozen graph and all model weights and the
model signature, if present in the meta data or inferred from the graph.

__ https://www.tensorflow.org/api_docs/python/tf/Graph
__ https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/saved_model/predict_signature_def


``graph_def_to_graph_v1``
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    graph_def_to_graph_v1(
        graph_def: GraphDef
    ) -> tf.Graph

..

Converts a ``GraphDef`` protobuf message to a ``tf.Graph``.

Use this function to convert the graph message loaded from a file to a
``tf.Graph`` that can be used for inference.

    **Arguments:**

**graph_def**
    GraphDef protobuf message, e.g. loaded from a file

..

    **Returns:**

The function returns a TF1 frozen ``tf.Graph`` that can be used for inference.

..

    **Example:**

.. code:: python

    from datetime import date

    import numpy as np
    import tensorflow as tf
    import tfjs_graph_converter.api as tfjs

    MODEL_PATH = './models/predict_lottery_numbers.pb'

    def load_frozen_graph(file_name):
        """Load a frozen graph from file and return protobuf message"""
        graph_def = tfjs.GraphDef()
        with open(file_name, 'rb') as proto_file:
            graph_def.ParseFromString(proto_file.read())
        return graph_def

    graph_def = load_frozen_graph(MODEL_PATH)
    graph = tfjs.graph_def_to_graph_v1(graph_def)
    # evaluate the loaded model
    with tf.compat.v1.Session(graph=graph) as sess:
        input_tensor_names = tfjs_util.get_input_tensors(graph)
        output_tensor_names = tfjs_util.get_output_tensors(graph)
        input_tensor = graph.get_tensor_by_name(input_tensor_names[0])
        today = date.today()
        vector = np.array([today.year, today.month, today.day],
                          dtype=np.float32)
        vector /= [2038, 12, 31]
        prediction = sess.run(output_tensor_names,
                           feed_dict={input_tensor: vector})
    # save the result
    prediction = prediction[0].numpy()
    print(f'Prediction for lottery numbers on {today}: {prediction}')


``graph_to_function_v2``
^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    graph_def_to_function_v2(
        graph: Union[tf.Graph, GraphDef]
    ) -> Callable

Wraps a GraphDef or TF1 frozen graph in a TF2 function for easy inference.

Use this function to convert a frozen graph returned by `load_graph_model`_
into a callable TF2 function.

The returned function will always take a single TF tensor as an input.
Multiple inputs can be used by placing them into the single tensor parameter:

.. code:: python

    import tfjs_graph_converter.api as tfjs

    graph_def = tfjs.load_graph_model('./models/some_tfjs_graph_model/')
    model = tfjs.graph_def_to_function_v2(graph_def)

    # 1st input: a 5-element vector
    input_1 = [1, 0, 2, 3, 0]
    # 2nd input: a 3x3 matrix
    input_2 = [[1, 0, 2], [1, 2, 0], [1, 5, 6]]
    # wrap inputs in a tf tensor
    inp = tf.constant([input_1, input_2])
    # evaluate f(input_1, input_2)
    predictions = model(inp)
    # result is a list of tensors that are the outputs of the model
    prediction = predictions[0]
    print(prediction.numpy())

..

    **Arguments:**

**graph**
    ``GraphDef`` protocol buffer message or TF1 frozen graph

..

    **Returns:**

The function returns a TF2 wrapped function that is callable with
input tensors as arguments and returns a list of model outputs as tensors.

..

    **Example:**

.. code:: python

    import numpy as np
    import tfjs_graph_converter.api as tfjs

    graph_def = tfjs.load_graph_model('./models/simple/')
    model = tfjs.graph_def_to_function_v2(graph_def)

    # extract a scalar from a tensor were tensor[np.argmax(tensor.shape)] == 1
    def as_scalar(tensor):
        array = tensor.numpy()
        flattened = np.reshape(array, (1))
        return flattened[0]

    # wrap scalar input in a tf tensor
    x = 16
    # model input has shape (1) and wrapped function expects a single tensor
    # that's the list of individual inputs, so from our scalar we get:
    inp = tf.constant([[x]])
    prediction = model(inp)
    # unpack scalar result: prediction is an array of tensors that are
    # the output of the model
    y = as_scalar(prediction[0])
    print(f'f({x}) = {y}')


``graph_model_to_frozen_graph``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   graph_model_to_frozen_graph(
        model_dir: str,
        export_path: str,
        compat_mode: CompatMode = CompatMode.NONE
   ) -> str

Converts a tensorflowjs graph model to a tensorflow frozen graph.
The resulting graph is written to a **binary** protobuf message.

..

    **Arguments:**

**model_dir**
    The directory that contains the ``model.json`` file.
    Alternatively, the path and name of the JSON file can be
    specified directly. Weight files must be located in the
    same directory as the model file.

**export_path**
    Directory and file name to save the frozen graph to.
    The file name usually ends in `.pb` and the directory
    must exist.

**compat_mode**
    Set this argument to the requested mode to ensure that the resulting graph
    is compatible with TensorflowJS or TFLite if possible.

    CompatMode.NONE: the resulting model uses the full set of TF ops

    CompatMode.TFJS: the resulting model uses TFJS-compatible data types

    CompatMode.TFJS: the resulting model uses TFLite builtin ops only

..

    **Returns:**

The returned string contains the location to which the frozen graph was
written.

..

    **Example:**

.. code:: python

   import tfjs_graph_converter.api as tfjs

   # convert TFJS model to a frozen graph
   tfjs.graph_model_to_frozen_graph(
        '~/some-website/saved_model_stylelize_js/',
        '~/models/stylize.pb')


``graph_model_to_saved_model``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   graph_model_to_saved_model(
        model_dir: str,
        export_dir: str,
        tags: Union[str, List[str]] = None,
        signature_def_map: dict = None,
        signature_key_map: RenameMap = None,
        compat_mode: CompatMode = CompatMode.NONE
   ) -> str

Converts a tensorflowjs graph model to a tensorflow `SavedModel`__
on disk. The functions reads and converts the graph model and saves it as a
`SavedModel` to the provided directory for further conversion or fine tuning.

__ https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md

..

    **Arguments:**

**model_dir**
    The directory that contains the ``model.json`` file.
    Alternatively, the path and name of the JSON file can be
    specified directly. Weight files must be located in the
    same directory as the model file.

**export_dir**
    Directory name to save the meta data and weights to.
    The directory must exist and should be empty.

**tags**
    List of strings that are annotations to identify the graph
    and its capabilities or purpose (e.g. ['serve']).
    Each meta graph added to the SavedModel must be annotated
    with user specified tags, which reflect the meta graph
    capabilities or use-cases. More specifically, these tags
    typically annotate a meta graph with its functionality
    (e.g. serving or training), and possibly hardware specific
    aspects such as GPU. Tags are optional and defaults apply if not provided.
    A single tag can be speficied as well.

**signature_def_map**
    Dict mapping signature keys (strings) to dict with the following supported
    (string-) keys:

    - ``"outputs"``: one or more outputs for this signature **required**
    - ``"method_name"``: method name if different from default *optional*

    Empty or `None` signature keys are replaced by the default signature key.

**signature_key_map**
    Optional mapping of tensor names to custom input or output names, see
    `RenameMap`_.

**compat_mode**
    Set this argument to the requested mode to ensure that the resulting graph
    is compatible with TensorflowJS or TFLite if possible.

    CompatMode.NONE: the resulting model uses the full set of TF ops

    CompatMode.TFJS: the resulting model uses TFJS-compatible data types

    CompatMode.TFJS: the resulting model uses TFLite builtin ops only

..

    **Returns:**

The returned string contains the location to which the meta graph and weights
were written.

..

    **Example:**

Export to a SavedModel using the default signature and -tags:

.. code:: python

   import tfjs_graph_converter as tfjs_conv

   tfjs_conv.api.graph_model_to_saved_model(
        '~/some-website/saved_model_stylelize_js/',
        '~/models/stylize/')

Export to a SavedModel using custom signatures (this example assumes a
multi-head model):

.. code:: python

    import tfjs_graph_converter as tfjs_conv
    from tfjs_graph_converter.api import RenameMap, SIGNATURE_OUTPUTS

    # here the model has two outputs - Identity and Identity_1, e.g. classify
    # scores and autoencoder output
    signature_map = {
        # add the default signature
        '': {SIGNATURE_OUTPUTS: 'Identity'},
        # add a generator signature
        'autoencode': {SIGNATURE_OUTPUTS: 'Identity_1'}
    }

    # rename the outputs to always be 'output'
    signature_key = RenameMap({
        'Identity': 'output', 'Identity_1': 'output', 'x': 'input'
    })

    tfjs_conv.api.graph_model_to_saved_model(
        '~/some-website/saved_model_stylelize_js/',
        '~/models/stylize/',
        signature_def_map=signature_map,
        signature_key_map=signature_key)


``graph_models_to_saved_model``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   graph_models_to_saved_model(
        model_list: List[Tuple[str, List[str]]],
        export_dir: str,
        signatures: dict = None,
        signature_keys: Dict[str, RenameMap] = None,
        compat_mode: CompatMode = CompatMode.NONE
    ) -> str


This function merges several tensorflowjs graph models into a single
`SavedModel`. Separate models are identified by different tags (see `documentation`__).

__ https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md

..

    **Arguments:**

**model_list**
    List of tuples containing the tensorflowjs graph model
    directory and a list of tags for the imported model.
    The content takes the form
    `[('/path/to/1st/model_json/', ['serve', 'preprocess']),`
    `('/path/to/2nd/model_json/', ['serve', 'predict'])]`

**export_dir**
    Directory name to save the meta data and weights to.
    The directory must exist and should be empty.

**signatures**
    Optional dict that maps model names (e.g. the first item in ``model_list``
    tuples) to signature dicts.
    The signature dict for each model maps signature keys to a list of outputs
    and an optional method name:

    - ``"outputs"``: one or more outputs for this signature **required**
    - ``"method_name"``: method name if different from default *optional*

    Empty or `None` keys are replaced with the default signature key.

**signature_key**
    Optional dict that maps model names (e.g. the first item of the tuples in
    ``model_list``) to `RenameMap`_ instances for assigning new names to model
    inputs and outputs.

**compat_mode**
    Set this argument to the requested mode to ensure that the resulting graph
    is compatible with TensorflowJS or TFLite if possible.

    CompatMode.NONE: the resulting model uses the full set of TF ops

    CompatMode.TFJS: the resulting model uses TFJS-compatible data types

    CompatMode.TFJS: the resulting model uses TFLite builtin ops only

..

    **Returns:**

The returned string contains the location to which the meta graph and weights
were written.

..

    **Example:**

The following example saves multiple models into a single SavedModel:

.. code:: python

    import tfjs_graph_converter.api as tfjs

    model_list = [
        ('~/website/preprocess_saved_model_js/', ['serve', 'preprocess']),
        ('~/website/predict_saved_model_js/', ['serve', 'predict']),
        ('~/website/finalize_saved_model_js/', ['serve', 'finalize'])
    ]
    # convert TFJS model to a SavedModel
    tfjs.graph_models_to_saved_model(model_list, '~/models/combined/')

The following example saves multiple models into a single SavedModel using
custom signatures:

.. code:: python

    import tfjs_graph_converter as tfjs_conv
    from tfjs_graph_converter.api import RenameMap, SIGNATURE_OUTPUTS

    model_list = [
        ('~/models/preprocess/', ['serve', 'preprocess']),
        ('~/models/predict/', ['serve', 'predict']),
        ('~/website/finalize/', ['serve', 'finalize'])
    ]
    # custom signatures for the first two models
    signatures = {
        '~/models/preprocess/': {
            '': {SIGNATURE_OUTPUTS: 'Identity'}
        },
        '~/models/predict/': {
            '': {SIGNATURE_OUTPUTS: 'Identity'}
            'generator': {SIGNATURE_OUTPUTS: 'Identity_1'}
        }
    }
    # rename the outputs to always be 'output'
    signature_keys = {
        '~/models/predict/': RenameMap({
            'Identity': 'output', 'Identity_1': 'output', 'x': 'input'
        }),
        '~/website/finalize/': RenameMap({
            'x': 'input', 'x_1': 'temperature', 'squeeze/dense4/BiasAdd': 'output'
        })
    }
    # convert TFJS model to a SavedModel
    tfjs_conv.api.graph_models_to_saved_model(
        model_list, '~/models/combined/',
        signatures=signatures,
        signature_keys=signature_keys)


``RenameMap``
^^^^^^^^^^^^^^^

.. code:: python

    RenameMap(
        mapping: Any,
    )

A ``RenameMap`` object is used for renaming inputs and outputs of a model
signature.

..

    **Arguments:**

**mapping**
    A ``dict`` that maps model input names (string keys) to new names
    (also strings) or any iterable that can be converted to a
    ``Dict[str, str]``.

All keys and values must be non-empty strings (whitespace-only is not allowed)
and all values (i.e. new names) don't have to be unique if the map is applied
to multiple signatures and doesn't cause name collisions.

..

    **Example:**

Let's pretend we have a multi-head model with two outputs: a one-hot classifier
result and an autoencoder output tensor. The default model signature contains
two outputs, `Identity` (the classifier result) and `Identity_1` (the
autoencoder output).

We want to set two signatures: one for the classifier result and one for the
autoencoder result. Both shall return their results in `output`.

.. code:: python

    import tfjs_graph_converter as tfjs_conv
    from tfjs_graph_converter.api import RenameMap

    # first we define out two signatures using the actual output names
    signature_map = {
        'serve/classify': {tfjs_conv.api.SIGNATURE_OUTPUTS: ['Identity']}
        'serve/autoencode': {tfjs_conv.api.SIGNATURE_OUTPUTS: ['Identity_1']}
    }
    # next we can define a RenameMap to change the keys of our outputs
    signature_key = RenameMap({'Identity': 'output', 'Identity_1': 'output'})

    # now we can convert our model to contain two signatures, both with a
    # single output called 'output':
    tfjs_conv.api.graph_model_to_saved_model(
        '~/models/multi-head/', '~/models/saved_model', tags=['serve'],
        signature_def_map=signature_map, signature_key_map=signature_key)


``RenameMap.apply``
^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    RenameMap.apply(
        signature: SignatureDef
    ) -> SignatureDef

This method applies the renaming to a given ``SignatureDef`` proto and returns
the updated signature.

..

    **Arguments:**

**signature**
    A ``SignatureDef`` proto containing the model with inputs or outputs to be
    renamed.

..

    **Returns:**

The updated ``SignatureDef`` proto containing the signature with inputs and
outputs renamed according to the map's contents.
