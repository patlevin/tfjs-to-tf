# SPDX-License-Identifier: MIT
# Copyright © 2020 Patrick Levin
"""Unit tests for graph optimization"""
from typing import List, Union
import unittest

import tensorflow as tf

from tfjs_graph_converter import optimization
import testutils


def _op_nodes(
        graph: Union[tf.Graph, testutils.GraphDef]) -> List[testutils.NodeDef]:
    """Return arithmetic/nn operation nodes from a graph"""
    graph_def = graph.as_graph_def() if isinstance(graph, tf.Graph) else graph
    def _op(node): return node.op not in ('Const', 'Identity', 'Placeholder')
    return [node for node in graph_def.node if _op(node)]


class OptimizationTest(unittest.TestCase):
    def test_optimize_graph(self):
        """optimize_graph should replace nodes if possible"""
        # generate optimisable test model
        input_graph = testutils.get_sample_graph()
        input_ops = [node.op for node in _op_nodes(input_graph)]
        # optimise the graph model
        output_graph = optimization.optimize_graph(input_graph)
        output_ops = [node.op for node in _op_nodes(output_graph)]
        # output should differ from input and be more efficient (smaller)
        self.assertNotEqual(input_ops, output_ops)
        self.assertLess(len(output_ops), len(input_ops))


if __name__ == '__main__':
    unittest.main()
