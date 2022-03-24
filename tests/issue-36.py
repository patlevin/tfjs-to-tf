# SPDX-License-Identifier: MIT
# Copyright © 2022 Patrick Levin
"""This script validates TFLite compatibility.

The script is used to replicate Github Issue #36
(https://github.com/patlevin/tfjs-to-tf/issues/36)

It expects the path to a converted blazeface model
(https://tfhub.dev/tensorflow/tfjs-model/blazeface/1/default/1?tfjs-format=compressed)
that was converted to a saved model, e.g. `tfjs_graph_converter
./models/blazeface ./models/blazeface_savedmodel --output_format tf_saved_model
`.

The script tries to load and convert the TF savedmodel to TFLite and reports
whether the conversion succeeded, e.g.:

`python issue-36.py ./models/blazeface_savedmodel`

Possible ouput formats are `FP32` (the default) and `INT8`

1) `python issue-36.py ./models/blazeface_savedmodel FP32`

2) `python issue-36.py ./models/blazeface_savedmodel INT8`

A model that was converted without the `--compat_mode=tflite` option will fail
to convert given INT8 (2)
"""
from enum import Enum
import os
import sys
import numpy as np
import tensorflow as tf


class Mode(Enum):
    FLOAT32 = 1
    QUANTISED = 2


def convert_to_tflite(savedmodel_path: str, mode: Mode):
    def representative_dummy_dataset():
        for _ in range(100):
            yield [
                np.zeros(128*128*3).reshape(1, 128, 128, 3).astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_saved_model(savedmodel_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dummy_dataset
    if mode == Mode.FLOAT32:
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    elif mode == Mode.QUANTISED:
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    else:
        raise Exception('Invalid conversion mode')
    tflite_model = converter.convert()
    return tflite_model


def main(args: list[str]):
    if len(args) < 2:
        print('Test conversion from TF saved_model to TFLite')
        print()
        print(f'Usage: {args[0]} <path_to_saved_model> [FP32|INT8]')
        exit(1)

    savedmodel_path = args[1]
    requested_mode = args[2].lower() if len(args) > 2 else 'fp32'
    if requested_mode == 'fp32':
        mode = Mode.FLOAT32
    elif requested_mode == 'int8':
        mode = Mode.QUANTISED
    else:
        print(f'Usage: {args[0]} <path_to_saved_model> [FP32|INT8]')
        exit(1)

    try:
        _ = convert_to_tflite(savedmodel_path, mode=mode)
    except Exception:
        print(f'CONVERSION FAILED: target mode={requested_mode}')
        exit(2)

    print(f'Conversion successful: target mode={requested_mode}')


if __name__ == '__main__':
    # reduce TF logging spam
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
    # disable CUDA, since conversion may fail if a CUDA-capable GPU is
    # installed that doesn't have the required capabilities or lacks VRAM
    os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
    main(sys.argv)
