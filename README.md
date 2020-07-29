# TensorFlow.js Graph Model Converter

![TFJS Graph Converter Logo](/docs/logo.png)

The purpose of this library is to import TFJS graph models into Tensorflow.
This allows you to use TensorFlow.js models with Python in case you don't
have access to the original formats or the models have been created in TFJS.

## Disclaimer

I'm neither a Python developer, nor do I know TensorFlow or TensorFlow.js.
I created this package solely because I ran into an issue when trying to convert
a pretrained TensorFlow.js model into a different format. I didn't have access to
the pretrained original TF model and didn't have the resources to train it myself.
I soon learned that I'm not alone with this [issue](https://github.com/tensorflow/tfjs/issues/1575)
so I sat down and wrote this little library.

If you find any part of the code to be non-idiomatic or know of a simpler way to
achieve certain things, feel free to let me know, since I'm a beginner in both
Python and especially TensorFlow (used it for the very first time in this
very project).

## Prerequisites

* tensorflow 2.1+
* tensorflowjs 1.5.2+

## Compatibility

The converter has been tested with tensorflowjs v1.7.2/v2.0.1 and tensorflow v2.1/v2.3.
The Python version used was Python 3.7.7.

## Installation

```sh
pip install tfjs-graph-converter
```

## Usage

After the installation, you can run the packaged `tfjs_graph_converter` binary
for quick and easy model conversion.

### Positional Arguments

 | Positional Argument | Description |
 | :--- | :--- |
 | `input_path` | Path to the TFJS Graph Model directory containing the model.json |
 | `output_path` | For output format "tf_saved_model", a SavedModel target directory. For output format "tf_frozen_model", a frozen model file. |

### Options

| Option | Description |
| :--- | :--- |
| `-h`, `--help` | Show help message and exit |
| `--output_format` | Use `tf_frozen_model` (the default) to save a Tensorflow frozen model. `tf_saved_model` exports to a Tensorflow _SavedModel_ instead. |
 | `--saved_model_tags` | Specifies the tags of the MetaGraphDef to save, in comma separated string format. Defaults to "serve". Applicable only if `--output format` is `tf_saved_model` |
 | `-v`, `--version` | Shows the version of the converter and its dependencies. |
 | `-s`, `--silent` | Suppresses any output besides error messages. |

Alternatively, you can create your own converter programs using the module's API.
The API is required to accomplish more complicated tasks, like packaging multiple
TensorFlow.js models into a single SavedModel.

## Example

To convert a TensorFlow.js graph model to a TensorFlow frozen model (i.e. the
most common use case?), just specify the directory containing the `model.json`,
followed by the path and file name of the frozen model like so:

```sh
tfjs_graph_converter path/to/js/model path/to/frozen/model.pb
```

## Usage from within Python

The package installs the module `tfjs_graph_converter`, which contains all the
functionality used by the converter script.
You can leverage the API to either load TensorFlow.js graph models directly for
use with your TensorFlow program (e.g. for inference, fine-tuning, or extending),
or use the advanced functionality to combine several TFJS models into a single
`SavedModel`.
The latter is only supported using the API (it's just a single function call,
though, so don't panic 😉)

[API Documentation](./docs/modules.rst)
