# TensorflowJS Graph Model Converter

The purpose of this library is to import TFJS graph models into Tensorflow.
This allows users to use TensorflowJS models with Python in case they don't
have access to the original formats.

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

* tensorflow 1.15.0+
* tensorflowjs 1.3.2+

## Compatibility

The converter has been tested with tensorflowjs v1.3.2 and tensorflow v1.15.0.
The Python version used was Python 3.7.5.

## Installation

To avoid version conflicts I recommend creating a new, clean Python environment.
Open a shell (or command window) and change to the repository's directory.
Example using [Anaconda](https://anaconda.org) on Windows, Linux or MacOSX:

```sh
conda create --name tfjs python=3.7
conda activate tfjs
pip install .
```

The tensorflowjs-package installed by pip comes with compatible versions of all
dependencies (including tensorflow). It is not necessary to install any special
tensorflow version (e.g. AVX2 or GPU-accelerated) for running the converter.

## Usage

After the installation, you can run the packaged `tfjs_graph_converter` binary
for quick and easy model conversion.
You can get a list of all supported options by using the _--help_ switch:

```sh
tfjs_graph_converter --help
```

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
though, so don't panic :D)

[API Documentation](./DOCUMENTATION.md)
