# SPDX-License-Identifier: MIT
# Copyright © 2020 Patrick Levin
"""Unit tests for the utility functions"""
import unittest

from tfjs_graph_converter import util
import testutils


class UtilTest(unittest.TestCase):

    def test_get_input_nodes(self):
        """Should return node info for inputs"""

        def _shape_of(node):
            shape = [d.size for d in node.attr['shape'].shape.dim]
            return [n if n > 0 else None for n in shape]

        graph = testutils.get_sample_graph()
        actual = util.get_input_nodes(graph)
        expected = testutils.get_inputs(graph.as_graph_def())
        self.assertEqual(len(actual), len(expected))
        for i, result in enumerate(actual):
            self.assertEqual(result.name, expected[i].name)
            self.assertEqual(result.shape, _shape_of(expected[i]))
            self.assertEqual(result.tensor, expected[i].name+':0')

    def test_get_output_nodes(self):
        """Should return node info for outputs"""
        graph_def = testutils.get_sample_graph_def()
        actual = util.get_output_nodes(graph_def)
        expected = testutils.get_outputs(graph_def)
        self.assertEqual(len(actual), len(expected))
        self.assertEqual(actual[0].name, expected[0].name)

    def test_get_input_tensors(self):
        """Should return tensor names for inputs"""
        graph_def = testutils.get_sample_graph_def()
        actual = util.get_input_tensors(graph_def)
        expected = [(n.name+':0') for n in testutils.get_inputs(graph_def)]
        self.assertEqual(actual, expected)

    def test_get_output_tensors(self):
        """Should return node info for outputs"""
        graph_def = testutils.get_sample_graph_def()
        actual = util.get_output_tensors(graph_def)
        expected = [(n.name+':0') for n in testutils.get_outputs(graph_def)]
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
