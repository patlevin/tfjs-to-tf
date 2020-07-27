# SPDX-License-Identifier: MIT
# Copyright © 2020 Patrick Levin
"""Unit tests for the public API functions"""
import os
import shutil
import tempfile
import unittest

import numpy as np
import tensorflow as tf

from tfjs_graph_converter import api
from tfjs_graph_converter import util
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.saved_model import constants

import testutils


def as_scalar(tensor_array: list) -> float:
    """Extract a scalar from a tensor that only contains a single value"""
    if len(tensor_array) > 1:
        raise ValueError(f'Expected scalar, got {tensor_array}')
    value = tensor_array[0].numpy()
    value = np.reshape(value, (1))
    return value[0]


def load_meta_graph(export_dir: str, tags: list
                    ) -> meta_graph_pb2.MetaGraphDef:
    path_to_pb = os.path.join(export_dir, constants.SAVED_MODEL_FILENAME_PB)
    # load from disk
    with tf.io.gfile.GFile(path_to_pb, 'rb') as f:
        saved_model = saved_model_pb2.SavedModel()
        saved_model.ParseFromString(f.read())
    # match tags
    if tags is None or len(tags) == 0:
        if len(saved_model.meta_graphs) == 1:
            return saved_model.meta_graphs[0]
        else:
            raise ValueError('tags are required - multiple meta graphs found')
    else:
        for meta_graph_def in saved_model.meta_graphs:
            if set(meta_graph_def.meta_info_def.tags) == set(tags):
                return meta_graph_def
        return None


class ApiTest(unittest.TestCase):
    def test_load_graph_model_with_simple_model(self):
        """load_graph_model should load simple model"""
        model_dir = testutils.get_path_to(testutils.SIMPLE_MODEL_PATH_NAME)
        graph = api.load_graph_model(model_dir)
        self.assertIsInstance(graph, tf.Graph)
        loaded_model = testutils.graph_to_model(graph)
        original_model_name = testutils.get_path_to(
            testutils.SIMPLE_MODEL_FILE_NAME)
        original_model = testutils.graph_to_model(original_model_name)
        # run both models and compare results
        x_ = 4
        x = tf.constant([[x_]], dtype=tf.float32)
        y_from_loaded_model = as_scalar(loaded_model(x))
        y_from_original_model = as_scalar(original_model(x))
        # sanity check; fails if model is different from the one we expected:
        # we want a model that predicts y = 5*x
        self.assertAlmostEqual(y_from_original_model, x_*5, places=1)
        # actual test
        self.assertAlmostEqual(y_from_loaded_model, y_from_original_model,
                               places=4)

    def test_load_graph_model_with_prelu(self):
        """load_graph_model should convert prelu operations"""
        model_dir = testutils.get_path_to(testutils.PRELU_MODEL_PATH)
        graph = api.load_graph_model(model_dir)
        loaded_model = testutils.graph_to_model(graph)
        original_model_name = testutils.get_path_to(
            testutils.PRELU_MODEL_FILE)
        original_model = testutils.graph_to_model(original_model_name)
        # run both models and compare results
        cx, cy, cz, r = -0.12, 0.2, 0.1, 0.314158
        px, py, pz = -0.4, 0.5, 0.4
        x = tf.constant([[cx, cy, cz, r, px, py, pz]], dtype=tf.float32)
        y_from_loaded_model = as_scalar(loaded_model(x))
        y_from_original_model = as_scalar(original_model(x))
        # sanity check; fails if model is different from the one we expected:
        # we want a model that predicts whether a point (px,py,pz) is inside
        # a sphere at (cx,cy,cz) of radius r
        self.assertAlmostEqual(y_from_original_model, 1, places=1)
        # actual test
        self.assertAlmostEqual(y_from_loaded_model, y_from_original_model,
                               places=4)

    def test_graph_def_to_graph_v1(self):
        """graph_def_to_graph_v1 should return tf.Graph for inference"""
        graph_def = testutils.get_sample_graph_def(
            testutils.SIMPLE_MODEL_FILE_NAME)
        graph = api.graph_def_to_graph_v1(graph_def)
        self.assertIsInstance(graph, tf.Graph)

    def test_graph_to_function_v2_given_graph_def(self):
        """graph_def_to_function_v2 should accept graph_def"""
        graph_def = testutils.get_sample_graph_def(
            testutils.SIMPLE_MODEL_FILE_NAME)
        estimate = api.graph_to_function_v2(graph_def)
        x_ = 20
        x = tf.constant([[x_]], dtype=tf.float32)
        y = as_scalar(estimate(x))
        self.assertAlmostEqual(y, x_*5, delta=0.1)

    def test_graph_to_function_v2_given_graph(self):
        """graph_def_to_function_v2 should accept tf.Graph"""
        graph = testutils.get_sample_graph(testutils.SIMPLE_MODEL_FILE_NAME)
        estimate = api.graph_to_function_v2(graph)
        x_ = 12
        x = tf.constant([[x_]], dtype=tf.float32)
        y = as_scalar(estimate(x))
        self.assertAlmostEqual(y, x_*5, places=1)

    def test_graph_model_to_frozen_graph(self):
        """graph_model_to_frozen_graph should save valid frozen graph model"""
        try:
            input_name = testutils.get_path_to(
                testutils.SIMPLE_MODEL_PATH_NAME)
            output_name = os.path.join(tempfile.gettempdir(), 'frozen.pb')
            api.graph_model_to_frozen_graph(input_name, output_name)
            # make sure the output file exists and isn't empty
            self.assertTrue(os.path.exists(output_name))
            self.assertGreater(os.stat(output_name).st_size, 256)
            # file must be a valid protobuf message
            with open(output_name, 'rb') as pb_file:
                graph_def = testutils.GraphDef()
                graph_def.ParseFromString(pb_file.read())
        finally:
            if os.path.exists(output_name):
                os.remove(output_name)

    def test_graph_model_to_saved_model(self):
        """graph_model_to_saved_model should save valid SavedModel"""
        model_dir = testutils.get_path_to(testutils.PRELU_MODEL_PATH)
        export_dir = tempfile.mkdtemp(suffix='.saved_model')
        try:
            tags = [tf.saved_model.SERVING]
            api.graph_model_to_saved_model(model_dir, export_dir,
                                           tags=tags)
            self.assertTrue(os.path.exists(export_dir))
            self.assertTrue(tf.saved_model.contains_saved_model(export_dir))
            # try and load the model
            meta_graph_def = load_meta_graph(export_dir, tags)
            self.assertIsNotNone(meta_graph_def)
            # we also want a signature to be present
            self.assertGreater(len(meta_graph_def.signature_def), 0)
            # the signatures should be valid
            for signature_def in meta_graph_def.signature_def.values():
                self.assertGreater(len(signature_def.inputs), 0)
                self.assertGreater(len(signature_def.outputs), 0)
                self.assertGreater(len(signature_def.method_name), 0)
        finally:
            if os.path.exists(export_dir):
                shutil.rmtree(export_dir)

    def test_graph_models_to_saved_model(self):
        """graph_models_to_saved_model should accept model list"""
        model_dir_1 = testutils.get_path_to(testutils.SIMPLE_MODEL_PATH_NAME)
        model_dir_2 = testutils.get_path_to(testutils.PRELU_MODEL_PATH)
        tags_1 = [tf.saved_model.SERVING, 'model_1']
        tags_2 = [tf.saved_model.SERVING, 'model_2']
        export_dir = tempfile.mkdtemp(suffix='.saved_model')
        try:
            api.graph_models_to_saved_model([
                                               (model_dir_1, tags_1),
                                               (model_dir_2, tags_2)
                                            ], export_dir)
            self.assertTrue(os.path.exists(export_dir))
            # try to load model 1
            meta_graph_def = load_meta_graph(export_dir, tags_1)
            self.assertIsNotNone(meta_graph_def)
            # we also want a signature to be present
            self.assertGreater(len(meta_graph_def.signature_def), 0)
            # the signatures should be valid
            for signature_def in meta_graph_def.signature_def.values():
                self.assertGreater(len(signature_def.inputs), 0)
                self.assertGreater(len(signature_def.outputs), 0)
                self.assertGreater(len(signature_def.method_name), 0)
            # try to load model 2
            meta_graph_def = load_meta_graph(export_dir, tags_2)
            self.assertIsNotNone(meta_graph_def)
            # we also want a signature to be present
            self.assertGreater(len(meta_graph_def.signature_def), 0)
            # the signatures should be valid
            for signature_def in meta_graph_def.signature_def.values():
                self.assertGreater(len(signature_def.inputs), 0)
                self.assertGreater(len(signature_def.outputs), 0)
                self.assertGreater(len(signature_def.method_name), 0)
        finally:
            if os.path.exists(export_dir):
                shutil.rmtree(export_dir)

    def test_load_graph_model_and_signature_from_meta_data(self):
        """load_graph_model_and_signature should extract signature def"""
        _, signature_def = api.load_graph_model_and_signature(
            testutils.get_path_to(testutils.PRELU_MODEL_PATH))
        self.assertIsInstance(signature_def, util.SignatureDef)
        self.assertGreater(len(signature_def.inputs), 0)
        self.assertGreater(len(signature_def.outputs), 0)
        self.assertGreater(len(signature_def.method_name), 0)

    def test_load_graph_model_and_signature_from_tree(self):
        """load_graph_model_and_signature should infer signature def
           from graph if signature def is incomplete
        """
        _, signature_def = api.load_graph_model_and_signature(
            testutils.get_path_to(testutils.SIMPLE_MODEL_PATH_NAME))
        # simple model is missing inputs in signature - defer from graph
        self.assertIsInstance(signature_def, util.SignatureDef)
        self.assertGreater(len(signature_def.inputs), 0)
        self.assertGreater(len(signature_def.outputs), 0)
        self.assertGreater(len(signature_def.method_name), 0)


if __name__ == '__main__':
    unittest.main()
