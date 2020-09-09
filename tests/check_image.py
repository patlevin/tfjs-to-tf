# SPDX-License-Identifier: MIT
# Copyright © 2020 Patrick Levin
"""Test script to check whether the classifier model works"""

import os
import sys
import numpy as np
import testutils as util
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import Resizing
import tfjs_graph_converter as tfjs

CLASS_NAME = ['horse', 'human']


def _evaluate(output):
    """Evaluate the model output and return class name and confidence"""
    y = output[0].numpy().reshape((2))
    index = np.argmax(y)
    name = CLASS_NAME[index]
    confidence = y[index]
    return (name, confidence)


if len(sys.argv) != 2:
    print(f"usage: {os.path.basename(sys.argv[0])} IMAGE")
    exit(1)

if not os.path.isfile(sys.argv[1]):
    print(f"{os.path.basename(sys.argv[0])}: '{sys.argv[1]}' is not a file")
    exit(1)

image = tf.keras.preprocessing.image.load_img(sys.argv[1])
image = tf.keras.preprocessing.image.img_to_array(image)
# normalise image data to [-1, 1]
image /= 127.5
image -= 1.
# ensure image size matches model input shape
if image.shape != (32, 32, 3):
    print(f"{os.path.basename(sys.argv[0])}: WARNING - image size "
          f"should be 32x32, not {image.shape[0]}x{image.shape[1]}")
    image = Resizing(height=32, width=32)(image)
# reshape to fit model input (and convert to tensor if necessary)
image = tf.reshape(image, [1, 32, 32, 3])
# grab the model file and convert graph to function
graph = util.get_sample_graph(util.get_path_to(util.DEPTHWISE_RELU_FILE))
model = tfjs.api.graph_to_function_v2(graph)
# run the model on the input image
result = model(image)
# show result
label, confidence = _evaluate(result)
print(f"Result: {label}, confidence={confidence}")
