# SPDX-License-Identifier: MIT
# Copyright © 2020 Patrick Levin
"""Unit tests for model fixing functions"""
import copy
import json
import unittest

from tfjs_graph_converter import quirks
import testutils


class QuirksTest(unittest.TestCase):
    def test_clean_model(self):
        """No fixes required - should result in a no-op"""
        clean_model_str = '{"node":[{"name":"input","op":"Placeholder",'     \
            + '"attr":{"shape":{"shape":{"dim":[{"size":"-1"},{"size":"28"},'\
            + '{"size":"28"},{"size":"1"}]}},"dtype":{"type":"DT_FLOAT"}}}]}'
        clean_model_json = json.loads(clean_model_str)
        expected = copy.deepcopy(clean_model_json)
        actual = quirks.fix_node_attributes(clean_model_json)
        self.assertEqual(actual, expected)

    def test_base64_conversion_from_ascii_codes(self):
        """Should convert string attr from ASCII codes to base64"""
        model_str = '{"node":[{"input":["MobilenetV2/Conv/Relu6",'      \
            + '"MobilenetV2/expanded_conv/depthwise/depthwise_weights"' \
            + '],"attr":{"padding":{"s":[83,65,77,69]},"dilations":{'   \
            + '"list":{"s":[],"i":["1","1","1","1"],"f":[],"b":[],'     \
            + '"type":[],"shape":[],"tensor":[],"func":[]}},"T":{'      \
            + '"type":1},"data_format":{"s":[78,72,87,67]},"strides":{' \
            + '"list":{"s":[],"i":["1","1","1","1"],"f":[],"b":[],'     \
            + '"type":[],"shape":[],"tensor":[],"func":[]}}},"name":'   \
            + '"MobilenetV2/expanded_conv/depthwise/depthwise",'        \
            + '"op": "DepthwiseConv2dNative"}]}'
        model_json = json.loads(model_str)
        quirks.fix_node_attributes(model_json)
        actual = testutils.select_all(model_json, 's')
        expected = [b'U0FNRQ==\n', None, b'TkhXQw==\n', None]
        self.assertEqual(actual, expected)

    def test_keep_properly_encoded_strings(self):
        """Should keep properly encoded string attr values"""
        model_str = '{"node":[{"name":"model/maxpool2d/MaxPool",'   \
            + '"op":"MaxPool","input":["model/conv2d/BiasAdd"],'    \
            + '"attr":{"ksize":{"list":{"i": ["1","2","2","1"]}},'  \
            + '"padding":{"s":"VkFMSUQ="},"T":{"type":"DT_FLOAT"},' \
            + '"strides":{"list":{"i": ["1","2","2","1"]}},'        \
            + '"data_format": {"s": "TkhXQw=="}}}]}'
        model_json = json.loads(model_str)
        quirks.fix_node_attributes(model_json)
        actual = testutils.select_all(model_json, 's')
        expected = ['VkFMSUQ=', 'TkhXQw==']
        self.assertEqual(actual, expected)

    def test_fix_dilations(self):
        """Should fix dilation attr values"""
        model_str = '{"node":[{"name":"resnet_v1_50/'                       \
            + 'block4/unit_3/bottleneck_v1/conv2/BatchNorm/batchnorm_1/'    \
            + 'add_1/conv","op":"Conv2D","input":["resnet_v1_50/block4/'    \
            + 'unit_3/bottleneck_v1/conv1/Relu","resnet_v1_50/block4/'      \
            + 'unit_3/bottleneck_v1/conv2/weights"],"attr":{"padding":'     \
            + '{"s":"U0FNRQ=="},"dilations":{"list":{"i":["2","2","1","1"]}'\
            + '}}}]}'
        model_json = json.loads(model_str)
        quirks.fix_node_attributes(model_json)
        actual = testutils.select_single(model_json, 'i')
        expected = ['1', '2', '2', '1']
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
