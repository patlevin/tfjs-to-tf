from __future__ import absolute_import

import os

import numpy as np
import tensorflow as tf
import tfjs_graph_converter as tfjs

STYLE_RATIO = 1.0# 0.75

style_model_dir = "../arbitrary-image-stylization-tfjs/saved_model_style_inception_js/"
style_model_name = "style_inception.pb"
transformer_model_dir = "../arbitrary-image-stylization-tfjs/saved_model_transformer_js/"
transformer_model_name = "transformer_inception.pb"
export_dir = "../models/"

style_image_path = "../examples/towers.jpg"
content_image_path = "../examples/chicago.jpg"
stylised_image_path = "../examples/stylised.jpg"

def get_target_size(shape):
    if len(shape) < 3 or len(shape) > 4:
        return None
    size = tuple(shape[len(shape)-3:-1])
    return None if None in size else size

# ported from JS - fromPixels(...).toFloat().div(tf.scalar(255)).expandDims()
def load_image(file_name, target_size=None, dtype=np.float32):
    img = tf.keras.preprocessing.image.load_img(file_name, target_size=target_size)
    x = tf.keras.preprocessing.image.img_to_array(img, dtype=dtype)
    x /= 255.0
    return x[tf.newaxis, ...]

def convert_model(model_dir, name, info=None):
    if info is None:
        info = model_dir

    file_name = os.path.join(export_dir, name)
    if not tf.io.gfile.exists(file_name):
        print("Converting {}...".format(info), end=" ")
        export_path = os.path.join(export_dir, name)
        tfjs.api.graph_model_to_frozen_graph(model_dir, export_path)
        print("[OK]")

    return

def convert_models():
    convert_model(style_model_dir, style_model_name, info="style model")
    convert_model(transformer_model_dir, transformer_model_name, info="transformer model")

    saved_model_dir = os.path.join(export_dir, 'style_inception')
    if not tf.io.gfile.exists(saved_model_dir):
        tfjs.api.graph_model_to_saved_model(style_model_dir, saved_model_dir, ['style'])

    saved_model_dir = os.path.join(export_dir, 'transformer_inception')
    if not tf.io.gfile.exists(saved_model_dir):
        tfjs.api.graph_model_to_saved_model(transformer_model_dir, saved_model_dir, ['transformer'])

    saved_model_dir = os.path.join(export_dir, 'inception')
    if not tf.io.gfile.exists(saved_model_dir):
        model_list = [
            (style_model_dir, ['style']), (transformer_model_dir, ['transformer'])
        ]
        tfjs.api.graph_models_to_saved_model(model_list, saved_model_dir)

    return

def load_model(name):
    file_name = os.path.join(export_dir, name)
    with tf.io.gfile.GFile(file_name, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    return graph_def

def load_models():
    print("Loading style model...", end=" ")
    style = load_model(style_model_name)
    print("[OK]")

    print("Loading transformer model...", end=" ")
    transformer = load_model(transformer_model_name)
    print("[OK]")

    return style, transformer

def load_input_images(graph_def):
    input_node = tfjs.util.get_input_nodes(graph_def)[0]
    input_size = get_target_size(input_node.shape)
    input_dtype = input_node.dtype

    print("Loading style image...", end=" ")
    style = load_image(style_image_path, target_size=input_size, dtype=input_dtype)
    print("OK")

    print("Loading content image...", end=" ")
    content = load_image(content_image_path, target_size=input_size, dtype=input_dtype)
    print("OK")

    return style, content

def load_saved_model(model_dir, tags):
    sess = tf.compat.v1.Session(graph=tf.Graph())
    tf.compat.v1.saved_model.loader.load(sess, tags, model_dir)
    return sess

def create_session(graph_def):
    with tf.Graph().as_default() as g:
        tf.import_graph_def(graph_def, name='')
    return tf.compat.v1.Session(graph=g)

def create_sessions(style, transformer):
    print("Creating sessions...", end=' ')
    style_sess = create_session(style)
    transformer_sess = create_session(transformer)
    print("[OK]")
    return style_sess, transformer_sess

def main_frozen():
    convert_models()
    style, transformer = load_models()
    style_image, content_image = load_input_images(style)
    style_sess, transformer_sess = create_sessions(style, transformer)

    print("Stylising image...", end=' ')

    style_outputs = tfjs.util.get_output_tensors(style)
    style_tensor_name = tfjs.util.get_input_tensors(style)[0]
    style_tensor = style_sess.graph.get_tensor_by_name(style_tensor_name)
    bottleneck = style_sess.run(style_outputs, feed_dict={style_tensor: style_image})[0]

    if STYLE_RATIO != 1.0:
        identity_bottleneck = style_sess.run(style_outputs, free_dict={style_tensor: content_image})[0]
        style_bottleneck_scaled = bottleneck * STYLE_RATIO
        identity_bottleneck_scaled = identity_bottleneck * (1.0 - STYLE_RATIO)
        bottleneck = style_bottleneck_scaled + identity_bottleneck_scaled

    transf_inputs = tfjs.util.get_input_tensors(transformer)
    transf_outputs = tfjs.util.get_output_tensors(transformer)
    content_tensor = transformer_sess.graph.get_tensor_by_name(transf_inputs[0])
    style_vector_tensor = transformer_sess.graph.get_tensor_by_name(transf_inputs[1])
    transformer_inputs = {
        content_tensor: content_image,
        style_vector_tensor: bottleneck
    }

    stylised = transformer_sess.run(transf_outputs, feed_dict=transformer_inputs)
    stylised = np.squeeze(stylised)
    print("[OK]")

    print("Saving result...", end=' ')
    tf.keras.preprocessing.image.save_img(stylised_image_path, stylised)
    print("[OK]")

def main_saved():
    convert_models()
    saved_model_dir = os.path.join(export_dir, 'inception')

    style_sess = load_saved_model(saved_model_dir, ['style'])
    transformer_sess = load_saved_model(saved_model_dir, ['transformer'])

    print("Loading input images...", end=' ')
    style_image = load_image(style_image_path)
    content_image = load_image(content_image_path)
    print("[OK]")

    print("Stylising image...", end=' ')
    style_input_tensor = tfjs.util.get_input_tensors(style_sess.graph)[0]
    style_outputs = tfjs.util.get_output_tensors(style_sess.graph)
    style_input = style_sess.graph.get_tensor_by_name(style_input_tensor)
    bottleneck = style_sess.run(style_outputs, feed_dict={style_input: style_image})[0]

    if STYLE_RATIO != 1.0:
        identity_bottleneck = style_sess.run(style_outputs, feed_dict={style_input: content_image})[0]
        style_bottleneck_scaled = bottleneck * STYLE_RATIO
        identity_bottleneck_scaled = identity_bottleneck * (1.0 - STYLE_RATIO)
        bottleneck = style_bottleneck_scaled + identity_bottleneck_scaled

    transformer_input_tensors = tfjs.util.get_input_tensors(transformer_sess.graph)
    transformer_outputs = tfjs.util.get_output_tensors(transformer_sess.graph)
    content_input = transformer_sess.graph.get_tensor_by_name(transformer_input_tensors[0])
    vector_input = transformer_sess.graph.get_tensor_by_name(transformer_input_tensors[1])
    transformer_inputs = {
        content_input: content_image,
        vector_input: bottleneck
    }
    stylised = transformer_sess.run(transformer_outputs, feed_dict=transformer_inputs)[0]
    stylised = np.squeeze(stylised)
    print("[OK]")

    print("Saving result...", end=' ')
    tf.keras.preprocessing.image.save_img(stylised_image_path, stylised)
    print("[OK]")

if __name__ == '__main__':
    main_frozen()
