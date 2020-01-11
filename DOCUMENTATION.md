# TensorFlow.js Graph Converter API

## Modules

The converter comes with several Python modules:

* **tfjs_graph_converter**
* [**tfjs_graph_converter.api**](#tfjs_graph_converter.api)
* [**tfjs_graph_converter.common**](#tfjs_graph_converter.common)
* **tfjs_graph_converter.converter**
* [**tfjs_graph_converter.util**](#tfjs_graph_converter.util)
* [**tfjs_graph_converter.version**](#tfjs_graph_converter.version)
* [**tfjs_graph_converter.common**](#tfjs_graph_converter.common)

## tfjs_graph_converter.api

Functions to load TensorFlow graphs from TensorFlow.js graph models and convert
TensorFlow.js graph models to TensorFlow frozen graph and SavedModel formats.

### Functions

* [load_graph_model](#tfjs_graph_converter.api.load_graph_model)
* [graph_model_to_frozen_graph](#tfjs_graph_converter.api.graph_model_to_frozen_graph)
* [graph_model_to_saved_model](#tfjs_graph_converter.api.graph_model_to_saved_model)
* [graph_models_to_saved_model](#tfjs_graph_converter.api.graph_models_to_saved_model)

### tfjs_graph_converter.api.load_graph_model

Load a tensorflowjs graph model from a directory and return a tensorflow (v1) graph.

#### Arguments

* **model_dir:** Name of the directory that contains the model.json and the weights

#### Returns

tensorflow [Graph](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/Graph)
object (frozen graph) that can be used for inference or converted into different
formats

#### Example

```python
import numpy as np
import tensorflow as tf
import tfjs_graph_converter as tfjs

# load TFJS model into tensorflow frozen graph
graph = tfjs.api.load_graph_model('~/some-website/saved_model_stylize_js/')

# load image into numpy array
img = tf.keras.preprocessing.image.load_img('~/images/sample.jpg')
x = tf.keras.preprocessing.image.img_to_array(img, dtype=np.float32)
# scale input to [0..1] and expand dimensions to 4d array
x /= 255.0
content_image = x[tf.newaxis, ...]

# evaluate the loaded model directly
with tf.compat.v1.Session(graph=graph) as sess:
    # the module provides some helpers for querying model properties
    input_tensor_names = tfjs.util.get_input_tensors(graph)
    output_tensor_names = tfjs.util.get_output_tensors(graph)
    input_tensor = graph.get_tensor_by_name(input_tensor_names[0])

    results = sess.run(output_tensor_names, feed_dict={input_tensor: content_image})

# save result
stylised = np.squeeze(results)
tf.keras.preprocessing.image.save_img('~/images/stylized_sample.jpg', stylised)
```

### tfjs_graph_converter.api.graph_model_to_frozen_graph

Convert a TensorFlow.js graph model to a tensorflow frozen graph.

#### Arguments

* **model_dir:** Name of the directory that contains the model.json and the weights
* **export_path:** Path to the frozen graph file

#### Example

```python
import tfjs_graph_converter as tfjs

# convert TFJS model to a frozen graph
tfjs.api.graph_model_to_frozen_graph(
    '~/some-website/saved_model_stylelize_js/', '~/models/stylize.pb')
```

### tfjs_graph_converter.api.graph_model_to_saved_model

Convert a TensorFlow.js graph model to a tensorflow [SavedModel](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md)
on disk.

#### Arguments

* **model_dir:** Name of the directory that contains the model.json and the weights
* **export_dir:** Directory to store the tensorflow SavedModel in
* **tags:** Array of strings that are annotations to identify the graph (e.g. ['serve'])

#### Example

```python
import tfjs_graph_converter as tfjs

# convert TFJS model to a SavedModel
tfjs.api.graph_model_to_saved_model('~/some-website/saved_model_stylelize_js/',
'~/models/stylize/', ['serve'])
```

### tfjs_graph_converter.api.graph_models_to_saved_model

Merge several TensorFlow.js graph models into a single SavedModel.
Separate models are identified by different tags (see [documentation](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md))

#### Arguments

* **model_list:** List of tuples containing the TensorFlow.js graph model
directory and a list of tags for the imported model. Format example:
`[('/path/to/1st/model_json/', ['serve', 'preprocess']), ('/path/to/2nd/model_json/',
['serve', 'predict'])]`
* **export_dir:** Directory to store the combined SavedModel in

#### Example

```python
import tfjs_graph_converter as tfjs

model_list = [
    ('~/website/preprocess_saved_model_js/', ['serve', 'preprocess']),
    ('~/website/predict_saved_model_js/', ['serve', 'predict']),
    ('~/website/finalize_saved_model_js/', ['serve', 'finalize'])
]
# convert TFJS model to a SavedModel
tfjs.api.graph_models_to_saved_model(model_list, '~/models/combined/')
```

## tfjs_graph_converter.util

Contains utility functions to query graph properties such as input- and output
tensor names.

### Contents

* [NodeInfo](#tfjs_graph_converter.util.NodeInfo)
* [get_input_nodes](#tfjs_graph_converter.util.get_input_nodes)
* [get_input_tensors](#fjs_graph_converter.util.get_input_tensors)
* [get_output_nodes](#tfjs_graph_converter.util.get_output_nodes)

### tfjs_graph_converter.util.NodeInfo

`namedtuple` with the following fields:

* **name:** Name of the graph node (e.g. layer name)
* **shape:** Shape associated with the tensor (input or output) as a list of sizes
(e.g. `[1, 256, 256, 3]`)
* **dtype:** _numpy_ data type of the tensor elements
* **tensor:** Name of the associated tensor for use with `graph.get_tensor_by_name()`

### tfjs_graph_converter.util.get_input_nodes

Return a list of `NodeInfo` tuples describing a graph's inputs.

#### Arguments

* **graph:** TF `Graph` or `GraphDef` object

#### Returns

List of `NodeInfo` tuples with basic properties of the graph's inputs.

#### Example

```python
import tensorflow as tf
import tfjs_graph_converter as tfjs

# load a frozen model graph, assume the input is a single image tensor
with tf.io.gfile.GFile("frozen_model.pb", "rb") as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

# get information about the input
model_input = tfjs.util.get_input_nodes(graph_def)[0]
input_shape = model_input.shape
# extract the width and height of the input dimension (assume "NHWC"-format)
target_size = input_shape[len(input_shape)-3,-1] if len(input_shape) in (3, 4) else None

# Load an input image according to the dimensions and element type of the model input
img = tf.keras.preprocessing.image.load_img("image.png", target_size=target_size)
x = tf.keras.preprocessing.image.img_to_array(img, dtype=model_input.dtype)
# scale pixels to [0..1]
x /= 255.0
# extend dimensions if required
if len(input_shape) > 3:
    x = x[tf.newaxis, ...]

# "x" is now a suitable input for the model
```

### tfjs_graph_converter.util.get_input_tensors

Return a list of input tensor names for use with `graph.get_tensor_by_name()`.

#### Arguments

* **graph:** TF `Graph` or `GraphDef` object

#### Returns

List of strings containing input tensor names.

#### Examples

```python
import tensorflow as tf
import tfjs_graph_converter as tfjs

# load a frozen model graph, assume the input is a single image tensor
with tf.io.gfile.GFile("frozen_model.pb", "rb") as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

# load suitable model input - see get_input_nodes() for an example
image_data = ...

# create a graph
with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def, name="")

# create a session
with tf.compat.v1.Session(graph=graph) as sess:
    output_tensors = tfjs.util.get_output_tensors(graph)
    input_name = tfjs.util.get_input_tensors(graph)[0]
    input_tensor = graph.get_tensor_by_name(input_name)
    # run the model
    results = sess.run(output_tensors, feed_dict={input_tensor: image_data})
# ...
```

### tfjs_graph_converter.util.get_output_nodes

Return a list of `NodeInfo` tuples describing a graph's outputs.

#### Arguments

* **graph:** TF `Graph` or `GraphDef` object

#### Returns

List of `NodeInfo` tuples with basic properties of the graph's outputs.
Note that the `shape` field is not being populated and remains an empty list.

### tfjs_graph_converter.util.get_output_tensors

Return a list of input tensor names for use with `graph.get_tensor_by_name()`.
This is useful for passing output tensor names to `Session.run()`.

#### Arguments

* **graph:** TF `Graph` or `GraphDef` object

#### Returns

List of strings containing output tensor names.

#### Example

* see `get_input_tensors()` for a usage example

## tfjs_graph_converter.version

Contains the module version.

### tfjs_graph_converter.version.VERSION

String that contains the module version.

## tfjs_graph_converter.common

Constants used throughout the module.
