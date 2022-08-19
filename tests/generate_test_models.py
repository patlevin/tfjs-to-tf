# SPDX-License-Identifier: MIT
# Copyright © 2020 Patrick Levin
"""Generates a models for unit testing"""

from pathlib import Path
from typing import Callable, Iterable
from zipfile import ZipFile
import os
import random
import tempfile
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, PReLU
from tensorflow.keras.layers import DepthwiseConv2D, Dropout, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from google.protobuf.json_format import MessageToJson
from tensorflowjs.converters.converter import convert as convert_to_tfjs
from tests.testutils import KERAS_MODEL_FILE_NAME
from tfjs_graph_converter.optimization import optimize_graph

from testutils import GraphDef, model_to_graph, get_outputs
from testutils import SAMPLE_MODEL_FILE_NAME, SIMPLE_MODEL_PATH_NAME
from testutils import PRELU_MODEL_PATH, MULTI_HEAD_PATH, DEPTHWISE_RELU_PATH
from testutils import IMAGE_DATASET, DEPTHWISE_PRELU_PATH
from testutils import get_path_to


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
    return Model(inp, out, name='model')


def simple_model():
    """Generate a single layer model that predicts y=5x"""
    inp = Input(shape=[1], name='x')
    out = Dense(1, name='output')(inp)
    model = Model(inp, out, name='simple_5x')
    model.compile(optimizer='sgd', loss='mean_squared_error')
    # train the model
    xs = np.concatenate((np.arange(17), [1, 5, 9]))
    np.random.shuffle(xs)
    ys = xs * 5
    print('Training the model... ', end='', flush=True)
    model.fit(xs, ys, epochs=500, verbose=0)
    print('done.')
    return model


def prelu_classifier_model():
    """
    Toy model that classifies whether a point is inside a sphere given the
    sphere's centre and radius as well as a sample point.
    """
    inp = Input(shape=[7], name='input_vector')   # cx,cy,cz,r,px,py,pz
    x = Dense(3, name='Dense')(inp)
    x = PReLU(name='Prelu')(x)
    out = Dense(1, activation='sigmoid', name='Output')(x)
    model = Model(inp, out, name='point_in_sphere')
    model.compile(optimizer='rmsprop', loss='binary_crossentropy',
                  metrics=['accuracy'])
    # generate 125 spheres of different sizes with sample 8 points per sphere,
    # half inside and half outside; use fixed rng seed for repeatability
    def gen_point(cx, cy, cz, r):
        u = np.random.random()
        v = np.random.random()
        theta = u * 2 * np.pi
        phi = np.arccos(2 * v - 1)
        sr = r * np.cbrt(np.random.random())
        st, ct = np.sin(theta), np.cos(theta)
        sp, cp = np.sin(phi), np.cos(phi)
        return cx + sr * sp * ct, cy + sr * sp * st, cz + sr * cp

    data = np.ndarray(shape=(1000, 8))
    np.random.seed(42)
    for s in range(125):
        cx = np.random.uniform(-1, 1)
        cy = np.random.uniform(-1, 1)
        cz = np.random.uniform(-1, 1)
        r = np.random.uniform(0.1, 1)
        # 4 samples inside the sphere
        for p in range(4):
            px, py, pz = gen_point(cx, cy, cz, r)
            assert (px-cx)*(px-cx) + (py-cy)*(py-cy) + (pz-cz)*(pz-cz) < r*r
            data[s*8+p] = [cx, cy, cz, r, px, py, pz, 1]
        # 4 samples outside the sphere
        for p in range(4):
            sr = -1 * r if random.random() < 0.5 else r
            px = cx + sr * random.uniform(1.02, 1.5)
            py = cy + sr * random.uniform(1.02, 1.5)
            pz = cz + sr * random.uniform(1.02, 1.5)
            assert (px-cx)*(px-cx) + (py-cy)*(py-cy) + (pz-cz)*(pz-cz) > r*r
            data[s*8+4+p] = [cx, cy, cz, r, px, py, pz, 0]

    np.random.shuffle(data)
    xs = data[:, 0:7]
    ys = data[:, 7]
    print('Training the model... ', end='', flush=True)
    model.fit(xs, ys, epochs=100, batch_size=32, verbose=0)
    print('done.')
    return model


def multi_head_model():
    # MNIST handwriting detection model
    inputs = Input(shape=(784,), name='image_data')
    dense1 = Dense(512, activation='relu')(inputs)
    dense2 = Dense(128, activation='relu')(dense1)
    dense3 = Dense(32, activation='relu')(dense2)
    # one-hot classifier output
    classification_output = Dense(10, activation='softmax',
                                  name='classification')(dense3)
    # upscale autoencoder output from bottleneck layer
    up_dense1 = Dense(128, activation='relu')(dense3)
    up_dense2 = Dense(512, activation='relu')(up_dense1)
    decoded_outputs = Dense(784, name='decoded_digit')(up_dense2)
    return Model(inputs, [classification_output, decoded_outputs])


def _load_hoh_dataset(tmpdirname):
    # load dataset from archive; return dataset tuple for training and testing
    archive = get_path_to(IMAGE_DATASET)
    with ZipFile(archive, 'r') as zip:
        zip.extractall(tmpdirname)
    path = os.path.join(tmpdirname, 'train')
    norm = Rescaling(1./127.5, offset=-1)
    train_ds = image_dataset_from_directory(path, image_size=(32, 32), seed=42)
    train_ds = train_ds.map(
        lambda x, y: (norm(x), tf.one_hot(y, depth=2)))
    path = os.path.join(tmpdirname, 'test')
    val_ds = image_dataset_from_directory(path, image_size=(32, 32), seed=42)
    val_ds = val_ds.map(
        lambda x, y: (norm(x), tf.one_hot(y, depth=2)))
    return (train_ds, val_ds)


def depthwise_model(activation: str = 'relu'):
    # image classification model (basically a stripped-down VGG16 variant)
    def _create_block(input_, activation=None, mode=None):
        # we need a bias initialiser to generate BiasAdd nodes
        bias_init = tf.keras.initializers.constant(.1)
        if mode == 'depthwise':
            if activation != 'prelu':
                x = DepthwiseConv2D(3, padding='same', use_bias=True,
                                    activation=activation,
                                    bias_initializer=bias_init)(input_)
            else:
                x = DepthwiseConv2D(3, padding='same', use_bias=True,
                                    bias_initializer=bias_init)(input_)
                x = PReLU()(x)
        else:
            x = Conv2D(32, 3, activation='relu')(input_)
        return MaxPooling2D(pool_size=(2, 2))(x)

    input_ = Input((32, 32, 3))
    x = _create_block(input_)
    x = _create_block(x)
    # these are just here for converter testing - a more efficient classifier
    # would use more Conv2D layers with increasing sizes
    x = _create_block(x, activation=activation, mode='depthwise')
    x = _create_block(x, activation=None, mode='depthwise')
    # two fully connected layers for classification
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.5, seed=23)(x)
    x = Dense(2, activation='softmax')(x)
    model = Model(input_, x)
    model.compile(optimizer='adam',
                  loss="categorical_crossentropy",
                  metrics=['categorical_accuracy'])
    # load training dataset (decompress to temp folder)
    with tempfile.TemporaryDirectory() as tmpdirname:
        random.seed(23)
        print('Loading dataset... ', end='', flush=True)
        train_ds, validate_ds = _load_hoh_dataset(tmpdirname)
        print('Ok.')
        # train for 15 epochs - should settle at about 91% accuracy
        print('Training the model... ', end='', flush=True)
        model.fit(train_ds, validation_data=validate_ds, epochs=15)
        print('Done')
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
    Path(path).mkdir(parents=True, exist_ok=True)
    model.save(os.path.join(path, 'keras.h5'))
    convert_to_tfjs([
        '--input_format=keras',
        '--output_format=tfjs_graph_model',
        os.path.join(path, 'keras.h5'), path
    ])


def save_layer_model(path: str) -> None:
    """Save empty TFJS layer model"""
    MODEL = (
        '{"modelTopology":{"class_name":"Sequential","config":[],' +
        '"keras_version":"tfjs-layers 1.1.2","backend":"tensor_flow.js"' +
        '},"format":"layers-model","generatedBy":null,"convertedBy":null,' +
        '"weightsManifest":[]}'
    )
    with open(path, 'w', encoding='utf-8') as f:
        f.write(MODEL)


if __name__ == '__main__':
    # print('Generating multi-layer sample model...')
    # model = deepmind_atari_net(10, input_shape=(128, 128, 3))
    # save_tf_model(model, get_path_to(SAMPLE_MODEL_FILE_NAME))
    # print('Generating single-layer simple model...')
    # model = simple_model()
    # save_tfjs_model(model, get_path_to(SIMPLE_MODEL_PATH_NAME))
    print('Generating prelu-activation model...')
    model = prelu_classifier_model()
    save_keras_model(model, get_path_to(PRELU_MODEL_PATH))
    # print('Generating multi-head model...')
    # model = multi_head_model()
    # save_tfjs_model(model, get_path_to(MULTI_HEAD_PATH))
    # print('Generating depthwise conv2d model...')
    # model = depthwise_model()
    # save_tfjs_model(model, get_path_to(DEPTHWISE_RELU_PATH))
    # print('Generating depthwise conv2d model with PReLU activation...')
    # model = depthwise_model('prelu')
    # save_tfjs_model(model, get_path_to(DEPTHWISE_PRELU_PATH))
    # print('Writing Keras layer model')
    # save_layer_model(get_path_to(KERAS_MODEL_FILE_NAME))