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
from tensorflow.compat.v1.saved_model import is_valid_signature
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


def _shape_of(tensor_info):
    return tuple(dim.size for dim in tensor_info.tensor_shape.dim)


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
            self.assertEqual(len(meta_graph_def.signature_def), 1)
            # the signatures should be valid
            self.assertTrue(is_valid_signature(
                list(meta_graph_def.signature_def.values())[0]
            ))
        finally:
            if os.path.exists(export_dir):
                shutil.rmtree(export_dir)

    def test_graph_model_to_saved_model_accepts_signature_map(self):
        """graph_model_to_saved_model should accept signature map"""
        model_dir = testutils.get_path_to(testutils.MULTI_HEAD_PATH)
        export_dir = tempfile.mkdtemp(suffix='.saved_model')
        try:
            tags = [tf.saved_model.SERVING]
            signature_map = {
                '': {api.SIGNATURE_OUTPUTS: ['Identity']},
                'debug': {api.SIGNATURE_OUTPUTS: ['Identity', 'Identity_1']}}
            api.graph_model_to_saved_model(model_dir, export_dir,
                                           tags=tags,
                                           signature_def_map=signature_map)
            # try and load the model
            meta_graph_def = load_meta_graph(export_dir, tags)
            self.assertIsNotNone(meta_graph_def)
            # we want both signatures to be present
            self.assertEqual(len(meta_graph_def.signature_def), 2)
            # the signatures should be valid
            for signature in meta_graph_def.signature_def.values():
                self.assertTrue(is_valid_signature(signature))
            # the default signature should have one output
            default_key = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY
            self.assertEqual(
                len(meta_graph_def.signature_def[default_key].outputs), 1)
            # debug signature should be present and contain two outputs
            self.assertIn('debug', meta_graph_def.signature_def.keys())
            self.assertEqual(
                len(meta_graph_def.signature_def['debug'].outputs), 2)
        finally:
            if os.path.exists(export_dir):
                shutil.rmtree(export_dir)

    def test_graph_model_to_saved_model_accepts_signature_key_map(self):
        """graph_model_to_saved_model should accept signature key map"""
        model_dir = testutils.get_path_to(testutils.MULTI_HEAD_PATH)
        export_dir = tempfile.mkdtemp(suffix='.saved_model')
        try:
            tags = [tf.saved_model.SERVING]
            signature_map = {
                '': {api.SIGNATURE_OUTPUTS: ['Identity']},
                'debug': {api.SIGNATURE_OUTPUTS: ['Identity', 'Identity_1']}}
            signature_key = api.RenameMap([
                ('Identity', 'output'), ('Identity_1', 'autoencoder_output')
            ])
            api.graph_model_to_saved_model(model_dir, export_dir,
                                           tags=tags,
                                           signature_def_map=signature_map,
                                           signature_key_map=signature_key)
            # try and load the model
            meta_graph_def = load_meta_graph(export_dir, tags)
            # the signatures should contain the renamed keys
            for signature in meta_graph_def.signature_def.values():
                self.assertIn('output', signature.outputs)
                self.assertEqual(signature.outputs['output'].name,
                                 'Identity:0')
            signature = meta_graph_def.signature_def['debug']
            self.assertIn('autoencoder_output', signature.outputs)
            self.assertEqual(signature.outputs['autoencoder_output'].name,
                             'Identity_1:0')
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
            self.assertEqual(len(meta_graph_def.signature_def), 1)
            # the signature should be valid
            self.assertTrue(is_valid_signature(
                list(meta_graph_def.signature_def.values())[0]
            ))
            # try to load model 2
            meta_graph_def = load_meta_graph(export_dir, tags_2)
            self.assertIsNotNone(meta_graph_def)
            # we also want a signature to be present
            self.assertEqual(len(meta_graph_def.signature_def), 1)
            # the signature should be valid
            self.assertTrue(is_valid_signature(
                list(meta_graph_def.signature_def.values())[0]
            ))
        finally:
            if os.path.exists(export_dir):
                shutil.rmtree(export_dir)

    def test_graph_models_to_saved_model_accepts_signatures(self):
        """graph_models_to_saved_model should accept signatures"""
        model_dir_1 = testutils.get_path_to(testutils.SIMPLE_MODEL_PATH_NAME)
        model_dir_2 = testutils.get_path_to(testutils.MULTI_HEAD_PATH)
        tags_1 = [tf.saved_model.SERVING, 'model_1']
        tags_2 = [tf.saved_model.SERVING, 'model_2']
        export_dir = tempfile.mkdtemp(suffix='.saved_model')
        try:
            model_list = [(model_dir_1, tags_1), (model_dir_2, tags_2)]
            signatures = {
                'ignore_this': {'': {api.SIGNATURE_OUTPUTS: ['y']}},
                model_dir_2: {'': {api.SIGNATURE_OUTPUTS: ['Identity']}}
            }
            api.graph_models_to_saved_model(model_list, export_dir, signatures)
            self.assertTrue(os.path.exists(export_dir))
            # try to load model 2
            meta_graph_def = load_meta_graph(export_dir, tags_2)
            # we want a signature to be present
            self.assertEqual(len(meta_graph_def.signature_def), 1)
            signature = list(meta_graph_def.signature_def.values())[0]
            # the signature should be valid
            self.assertTrue(is_valid_signature(signature))
            # the signature should have a single output
            self.assertEqual(len(signature.outputs), 1)
            self.assertIn('Identity', signature.outputs.keys())
        finally:
            if os.path.exists(export_dir):
                shutil.rmtree(export_dir)

    def test_graph_models_to_saved_model_accepts_signature_keys(self):
        """graph_models_to_saved_model should accept signature keys"""
        model_dir_1 = testutils.get_path_to(testutils.SIMPLE_MODEL_PATH_NAME)
        model_dir_2 = testutils.get_path_to(testutils.MULTI_HEAD_PATH)
        tags_1 = [tf.saved_model.SERVING, 'model_1']
        tags_2 = [tf.saved_model.SERVING, 'model_2']
        export_dir = tempfile.mkdtemp(suffix='.saved_model')
        try:
            model_list = [(model_dir_1, tags_1), (model_dir_2, tags_2)]
            signatures = {
                'ignore_this': {'': {api.SIGNATURE_OUTPUTS: ['y']}},
                model_dir_2: {'': {api.SIGNATURE_OUTPUTS: ['Identity']}}
            }
            signature_keys = {
                model_dir_1: api.RenameMap(
                    {'x': 'input', 'Identity': 'output'}
                ),
                model_dir_2: api.RenameMap({'Identity': 'scores'})
            }
            api.graph_models_to_saved_model(model_list, export_dir,
                                            signatures, signature_keys)
            self.assertTrue(os.path.exists(export_dir))
            # check the signatures of model 1
            meta_graph_def = load_meta_graph(export_dir, tags_1)
            signature = list(meta_graph_def.signature_def.values())[0]
            self.assertIn('input', signature.inputs)
            self.assertEqual(signature.inputs['input'].name, 'x:0')
            self.assertIn('output', signature.outputs)
            self.assertEqual(signature.outputs['output'].name, 'Identity:0')
            # check the signatures of model 2
            meta_graph_def = load_meta_graph(export_dir, tags_2)
            signature = list(meta_graph_def.signature_def.values())[0]
            self.assertIn('scores', signature.outputs)
            self.assertEqual(signature.outputs['scores'].name, 'Identity:0')
        finally:
            if os.path.exists(export_dir):
                shutil.rmtree(export_dir)

    def test_load_graph_model_and_signature_from_meta_data(self):
        """load_graph_model_and_signature should extract signature def"""
        _, signature_def = api.load_graph_model_and_signature(
            testutils.get_path_to(testutils.PRELU_MODEL_PATH))
        self.assertIsInstance(signature_def, util.SignatureDef)
        self.assertTrue(is_valid_signature(signature_def))
        self.assertEqual(len(signature_def.inputs), 1)
        key, value = list(signature_def.inputs.items())[0]
        self.assertEqual(key, 'input_vector')
        self.assertEqual(value.name, 'input_vector:0')
        self.assertEqual(value.dtype, tf.dtypes.float32)
        self.assertEqual(_shape_of(value), (-1, 7))
        self.assertEqual(len(signature_def.outputs), 1)
        key, value = list(signature_def.outputs.items())[0]
        self.assertEqual(key, 'Identity')
        self.assertEqual(value.name, 'Identity:0')
        self.assertEqual(value.dtype, tf.dtypes.float32)
        self.assertEqual(_shape_of(value), (-1, 1))

    def test_load_graph_model_and_signature_from_tree(self):
        """load_graph_model_and_signature should infer signature def
           from graph if signature def is incomplete
        """
        _, signature_def = api.load_graph_model_and_signature(
            testutils.get_path_to(testutils.SIMPLE_MODEL_PATH_NAME))
        # simple model is missing inputs in signature - defer from graph
        self.assertIsInstance(signature_def, util.SignatureDef)
        self.assertTrue(is_valid_signature(signature_def))
        self.assertEqual(len(signature_def.inputs), 1)
        key, value = list(signature_def.inputs.items())[0]
        self.assertEqual(key, 'x')
        self.assertEqual(value.name, 'x:0')
        self.assertEqual(value.dtype, tf.dtypes.float32)
        self.assertEqual(_shape_of(value), (-1, 1))
        self.assertEqual(len(signature_def.outputs), 1)
        key, value = list(signature_def.outputs.items())[0]
        self.assertEqual(key, 'Identity')
        self.assertEqual(value.name, 'Identity:0')
        self.assertEqual(value.dtype, tf.dtypes.float32)
        self.assertEqual(_shape_of(value), (-1, 1))

    def test_build_signatures_applies_defaults(self):
        """_build_signatures should return defaults given None or empty"""
        graph = testutils.get_sample_graph()
        signature_map = {None: {api.SIGNATURE_OUTPUTS: ['Identity']}}
        signature_def_map = api._build_signatures(graph, signature_map)
        default_key = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        self.assertIn(default_key, signature_def_map)
        default_name = tf.saved_model.PREDICT_METHOD_NAME
        self.assertEqual(signature_def_map[default_key].method_name,
                         default_name)
        # empty method names map to default, too
        signature_map = {None: {api.SIGNATURE_OUTPUTS: ['Identity'],
                                api.SIGNATURE_METHOD: ''}}
        signature_def_map = api._build_signatures(graph, signature_map)
        self.assertEqual(signature_def_map[default_key].method_name,
                         default_name)

    def test_build_signatures(self):
        """_build_signatures should apply given key and include inputs"""
        graph = testutils.get_sample_graph(
            testutils.get_path_to(testutils.MULTI_HEAD_FILE))
        default_key = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        debug_key = 'debug_model'
        signature_map = {
            '': {api.SIGNATURE_OUTPUTS: ['Identity']},
            debug_key: {api.SIGNATURE_OUTPUTS: ['Identity', 'Identity_1']}}

        signature_def_map = api._build_signatures(graph, signature_map)
        self.assertIn(default_key, signature_def_map)
        self.assertIn(debug_key, signature_def_map)
        for signature_def in signature_def_map.values():
            self.assertTrue(is_valid_signature(signature_def))

    def test_build_signatures_verifies_outputs(self):
        """_build_signatures should not accept invalid output names"""
        graph = testutils.get_sample_graph()
        signature_map = {'': {api.SIGNATURE_OUTPUTS: ['Not_A_Tensor']}}
        self.assertRaises(ValueError,
                          lambda: api._build_signatures(graph, signature_map))

    def test_rename_map_ctor_empty(self):
        """RenameMap constructor accepts empty arguments"""
        rename_map = api.RenameMap({})
        self.assertTrue(not any(rename_map.mapping))
        rename_map = api.RenameMap(dict())
        self.assertTrue(not any(rename_map.mapping))

    def test_rename_map_ctor_verifies_input_format(self):
        """RenameMap constructor verifies argument types"""
        # pass non-dictionary
        self.assertRaises(ValueError, lambda: api.RenameMap(5))
        # key is not a str
        self.assertRaises(ValueError, lambda: api.RenameMap({
            'valid': 'ok',
            True: 'that cannot work'
        }))
        # value is not a str
        self.assertRaises(ValueError, lambda: api.RenameMap({
            'valid': 'ok',
            'no-go': ['invalid']
        }))
        # empty key
        self.assertRaises(ValueError, lambda: api.RenameMap({
            'valid': 'ok',
            '': 'invalid'
        }))
        # empty value
        self.assertRaises(ValueError, lambda: api.RenameMap({
            'valid': 'ok',
            'invalid': '\n  \r\t'
        }))

    def test_rename_map_ctor_accepts_str_to_str_dict(self):
        """RenameMap constructor accepts Dict[str, str] and iterable"""
        rename_map = api.RenameMap(
            {'Identity': 'stylised_image'}
        )
        self.assertEqual(rename_map.mapping['Identity'], 'stylised_image')
        rename_map = api.RenameMap(
            [('Identity', 'stylised_image'), ('input', 'original_image')]
        )
        self.assertEqual(rename_map.mapping['Identity'], 'stylised_image')
        self.assertEqual(rename_map.mapping['input'], 'original_image')

    def test_rename_map_apply_requires_signature_def(self):
        """RenameMap.apply accepts SignatureDef only"""
        self.assertRaises(ValueError, lambda: api.RenameMap({}).apply(5))

    def test_rename_map_apply(self):
        """RenameMap.apply maps old names to new names"""
        _, signature_def = api.load_graph_model_and_signature(
            testutils.get_path_to(testutils.SIMPLE_MODEL_PATH_NAME))
        mapping = api.RenameMap({'x': 'input', 'Identity': 'output'})
        updated = mapping.apply(signature_def)
        self.assertNotIn('x', updated.inputs)
        self.assertIn('input', updated.inputs)
        # keep the tensor name!
        self.assertEqual(updated.inputs['input'].name, 'x:0')
        self.assertNotIn('Identity', updated.outputs)
        self.assertIn('output', updated.outputs)
        # keep the tensor name!
        self.assertEqual(updated.outputs['output'].name, 'Identity:0')


if __name__ == '__main__':
    unittest.main()
