# SPDX-License-Identifier: MIT
# Copyright © 2020 Patrick Levin
"""Generates a models for unit testing"""

from typing import Callable, Iterable
import os
import random

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, PReLU
from tensorflow.keras.models import Model
from google.protobuf.json_format import MessageToJson
from tensorflowjs.converters.converter import convert as convert_to_tfjs

from tfjs_graph_converter.optimization import optimize_graph

from testutils import GraphDef, model_to_graph, get_outputs
from testutils import SAMPLE_MODEL_FILE_NAME, SIMPLE_MODEL_PATH_NAME
from testutils import PRELU_MODEL_PATH


def deepmind_atari_net(num_classes: int = 10,
                       input_shape: Iterable[int] = (128, 128)) -> Model:
    """Generate optimisable test model

        Test model features multiple convolution layers with activation.
        This translates to sub-graphs [Conv2D, BiasAdd, Activation] per
        convolution-layer.

        Grappler should convert variables to constants and get rid of the
        BiasAdd.

        The two dense layers at the bottom translate to
        [MatMul, BiasAdd, Relu|Softmax]. Here, grappler should be able to
        remove the BiasAdd just like with the convolution layers.
    """
    inp = Input(shape=input_shape)
    x = Conv2D(32, kernel_size=(8, 8), strides=(4, 4), padding='same',
               activation='relu', name='conv1')(inp)
    x = Conv2D(64, kernel_size=(4, 4), strides=(2, 2), padding='same',
               activation='relu', name='conv2')(x)
    x = Conv2D(64, kernel_size=(3, 3), padding='same',
               activation='relu', name='conv3')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(512, activation='relu', name='dense1')(x)
    out = Dense(num_classes, activation='softmax', name='output')(x)
    return Model(inp, out)


def simple_model():
    """Generate a single layer model that predicts y=5x"""
    inp = Input(shape=[1])
    out = Dense(1, name='output')(inp)
    model = Model(inp, out, name='simple_5x')
    model.compile(optimizer='sgd', loss='mean_squared_error')
    # train the model
    xs = np.concatenate((np.arange(10), [1, 5, 9]))
    np.random.shuffle(xs)
    ys = xs * 5
    print('Training the model... ', end='')
    model.fit(xs, ys, epochs=500, verbose=0)
    print('done.')
    return model


def prelu_classifier_model():
    """
    Toy model that classifies whether a point is inside a sphere given the
    sphere's centre and radius as well as a sample point.
    """
    inp = Input(shape=[7])   # cx,cy,cz,r,px,py,pz
    x = Dense(3, name='Dense')(inp)
    x = PReLU(name='Prelu')(x)
    out = Dense(1, activation='sigmoid', name='Output')(x)
    model = Model(inp, out, name='point_in_sphere')
    model.compile(optimizer='rmsprop', loss='binary_crossentropy',
                  metrics=['accuracy'])
    # generate 125 spheres of different sizes with sample 8 points per sphere,
    # half inside and half outside; use fixed rng seed for repeatability
    data = np.ndarray(shape=(1000, 8))
    random.seed(42)
    for s in range(125):
        cx = random.uniform(-1, 1)
        cy = random.uniform(-1, 1)
        cz = random.uniform(-1, 1)
        r = random.uniform(0.01, 0.6)
        # 4 samples inside the sphere
        for p in range(4):
            px = cx + random.uniform(-r*0.9, r*0.9)
            py = cy + random.uniform(-r*0.9, r*0.9)
            pz = cz + random.uniform(-r*0.9, r*0.9)
            data[s*8+p] = [cx, cy, cz, r, px, py, pz, 1]
        # 4 samples outside the sphere
        for p in range(4):
            sr = -1 * r if random.random() < 0.5 else r
            px = cx + sr * random.uniform(1.02, 1.5)
            py = cy + sr * random.uniform(1.02, 1.5)
            pz = cz + sr * random.uniform(1.02, 1.5)
            data[s*8+4+p] = [cx, cy, cz, r, px, py, pz, 0]
    np.random.shuffle(data)
    xs = data[:, 0:7]
    ys = data[:, 7]
    print('Training the model... ', end='')
    model.fit(xs, ys, epochs=250, batch_size=32, verbose=0)
    print('done.')
    return model


def remove_weight_data(graph_def: GraphDef) -> None:
    """Remove dummy weight data from graph"""
    def _used_by(node_name, op_name):
        for node in graph_def.node:
            if node_name in node.input and node.op == op_name:
                return True
        return False

    for node in graph_def.node:
        if node.op == 'Const' and not _used_by(node.name, 'Reshape'):
            node.attr['value'].tensor.tensor_content = b''


def save_tf_model(model: Callable, path: str) -> None:
    """Save Keras model as frozen graph formatted as JSON"""
    graph = model_to_graph(model)
    graph_def = graph.as_graph_def()
    remove_weight_data(graph_def)
    message_json = MessageToJson(graph_def)
    with open(path, 'w') as f:
        f.write(message_json)


def save_tfjs_model(model: Callable, path: str) -> None:
    """Save Keras model as TFJS graph model"""
    graph = model_to_graph(model)
    graph_def = optimize_graph(graph)
    outputs = ','.join([node.name for node in get_outputs(graph_def)])
    tf.io.write_graph(graph_def, path, 'frozen_graph.pb', as_text=False)
    convert_to_tfjs([
        '--input_format=tf_frozen_model',
        '--output_format=tfjs_graph_model',
        f'--output_node_names={outputs}',
        os.path.join(path, 'frozen_graph.pb'), path
    ])


def save_keras_model(model: Callable, path: str) -> None:
    """Save Keras model as TFJS graph model"""
    model.save(os.path.join(path, 'keras.h5'))
    convert_to_tfjs([
        '--input_format=keras',
        '--output_format=tfjs_graph_model',
        os.path.join(path, 'keras.h5'), path
    ])


if __name__ == '__main__':
    print('Generating multi-layer sample model...')
    model = deepmind_atari_net(10, input_shape=(128, 128, 3))
    save_tf_model(model, SAMPLE_MODEL_FILE_NAME)
    print('Generating single-layer simple model...')
    model = simple_model()
    save_tfjs_model(model, SIMPLE_MODEL_PATH_NAME)
    print('Generating prelu-activation model...')
    model = prelu_classifier_model()
    save_keras_model(model, PRELU_MODEL_PATH)
