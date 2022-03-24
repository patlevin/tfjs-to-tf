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

The converter has been tested with tensorflowjs v3.13.0, tensorflow v2.8.0
and Python 3.9.10.

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
| `--saved_model_tags` | Specifies the tags of the MetaGraphDef to save, in comma separated string format. Defaults to "serve". Applicable only if `--output_format` is `tf_saved_model` |
| `-c`, `--compat_mode` | Sets a compatibility mode forthe coverted model (see below) |
| `-v`, `--version` | Shows the version of the converter and its dependencies. |
| `-s`, `--silent` | Suppresses any output besides error messages. |

### Compatibility Modes

Models are converted to optmimised native Tensorflow operators by default.
This can cause problems if the converted model is subseuently converted to
another format (ONNX, TFLite, older TFJS versions, etc.) The `--compat_mode`
option can be used to avoid incompatible native operations such as fused
convolutions. Available options are:

| Mode     | Description |
| :---     | :---        |
| `none`   | Use all available optimisations and native TF operators |
| `tfjs`   | Harmonise input types for compatibility with older TFJS versions |
| `tflite` | Only use TFLite builtins in the converted model |

### Advanced Options

These options are intended for advanced users who are familiar with the details of TensorFlow and [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving).

| Option | Description | Example |
| :--- | :--- | :--- |
| `--outputs` | Specifies the outputs of the MetaGraphDef to save, in comma separated string format. Applicable only if `--output_format` is `tf_saved_model` | --outputs=Identity |
| `--signature_key` | Specifies the key for the signature of the MetraGraphDef. Applicable only if `--output_format` is `tf_saved_model`. Requires `--outputs` to be set. | --signature_key=serving_autoencode |
| `--method_name` | Specifies the method name for the signature of the MetraGraphDef. Applicable only if `--output_format` is `tf_saved_model`. Requires `--outputs` to be set. | --method_name=tensorflow/serving/classify |
| `--rename` | Specifies a key mapping to change the keys of outputs and inputs in the signature. The format is comma-separated pairs of _old name:new name_. Applicable only if `--output_format` is `tf_saved_model`. Requires `--outputs` to be set. | --rename Identity:scores,model/dense256/BiasAdd:confidence |

Specifying ``--outputs`` can be useful for multi-head models to select the default
output for the main signature. The CLI only handles the default signature of
the model. Multiple signatures can be created using the [API](https://github.com/patlevin/tfjs-to-tf/blob/master/docs/api.rst).

The method name must be handled with care, since setting the wrong value might
prevent the signature from being valid for use with TensorFlow Serving.
The option is available, because the converter only generates
_predict_-signatures. In case the model is a regression model or a classifier
with the matching outputs, the correct method name can be forced using the
``--method_name`` option.

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

## Advanced Example

Converting to [TF SavedMovel format](https://www.tensorflow.org/guide/saved_model)
adds a lot of options for tweaking model signatures. The following example
converts a [Posenet](https://github.com/tensorflow/tfjs-models/tree/master/posenet)
model, which is a multi-head model.

We want to select only two of the four possible outputs and rename them in the
model's signature, as follows:

* Input: _input_ (from _sub_2_)
* Outputs: _offsets_ and _heatmaps_ (from _float_short_offsets_ and _float_heatmaps_)

```sh
tfjs_graph_converter \
    ~/models/posenet/model-stride16.json \
    ~/models/posenet_savedmodel \
    --output_format tf_saved_model \
    --outputs float_short_offsets,float_heatmaps \
    --rename float_short_offsets:offsets,float_heatmaps:heatmaps,sub_2:input
```

After the conversion, we can examine the output and verify the new model
signature:

```sh
saved_model_cli show --dir ~/models/posenet_savedmodel --all

MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['input'] tensor_info:
        dtype: DT_FLOAT
        shape: (1, -1, -1, 3)
        name: sub_2:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['heatmaps'] tensor_info:
        dtype: DT_FLOAT
        shape: (1, -1, -1, 17)
        name: float_heatmaps:0
    outputs['offsets'] tensor_info:
        dtype: DT_FLOAT
        shape: (1, -1, -1, 34)
        name: float_short_offsets:0
  Method name is: tensorflow/serving/predict
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

## Important Note

By default, Python code that includes the library will see CUDA devices
disabled (i.e. not visible in the program). This is done because the library
uses some low-level APIs that don't allow disabling GPUs from Python code.
Unfortunately some GPUs support CUDA but don't have the compute capabilities or
VRAM required to convert certain models. For this reason, CUDA devices are
disabled by default and the converter and scripts using it use CPUs only.

This behaviour can now be disabled by calling `enable_cuda()` **before** any
Tensorflow or converter function is called. This will re-enable the use of
CUDA-capable devices, but may result in errors during model
loading/conversion depending on the installed GPU hardware.

[API Documentation](./docs/modules.rst)
