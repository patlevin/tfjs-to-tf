# SPDX-License-Identifier: MIT
# Copyright © 2020 Patrick Levin
"""Unit tests for graph rewriting utilities"""
import unittest

import numpy as np
from tfjs_graph_converter import graph_rewrite_util as rewrite
import testutils


def _get_node_by_name(graph_def: rewrite.GraphDef,
                      node_name: str) -> rewrite.NodeDef:
    """Return a node from a graph that matches the provided name"""
    matches = [node for node in graph_def.node if node.name == node_name]
    return matches[0] if len(matches) > 0 else None


class GraphRewriteUtilTest(unittest.TestCase):
    def test_get_op_def_given_known_op(self):
        """Should return valid op def for known operations"""
        op_def = rewrite.get_op_def('MatMul')
        self.assertIsNotNone(op_def)

    def test_get_op_def_given_bogus_op(self):
        """Should return None for unknown operations"""
        op_def = rewrite.get_op_def('CureCancer')
        self.assertIsNone(op_def)

    def test_make_op_node_given_str_inputs(self):
        """make_op_node should support list of strings for inputs"""
        node = rewrite.make_op_node('Add', ['x', 'y'])
        self.assertEqual(node.name, 'Add')
        self.assertEqual(node.op, 'Add')
        self.assertEqual(node.input, ['x', 'y'])

    def test_make_op_node_given_node_inputs(self):
        """make_op_node should support list of nodes for inputs"""
        node_a = rewrite.make_op_node('Neg', ['x'], name='node_a')
        node_b = rewrite.make_op_node('Identity', ['Neg'], name='node_b')
        node = rewrite.make_op_node('Add', [node_a, node_b])
        self.assertEqual(node.name, 'Add')
        self.assertEqual(node.op, 'Add')
        self.assertEqual(node.input, ['node_a', 'node_b'])

    def test_make_op_node_given_mixed_inputs(self):
        """make_op_node should support list of mixed types for inputs"""
        node_a = rewrite.make_op_node('Neg', ['x'], name='node_a')
        node = rewrite.make_op_node('Add', [node_a, 'node_b'])
        self.assertEqual(node.name, 'Add')
        self.assertEqual(node.op, 'Add')
        self.assertEqual(node.input, ['node_a', 'node_b'])

    def test_make_op_node_given_input_as_scalar(self):
        """make_op_node should accept scalar input of type string and node"""
        node_from_str = rewrite.make_op_node('Neg', 'x', name='node_a')
        self.assertEqual(node_from_str.input, ['x'])
        node_from_node = rewrite.make_op_node('Neg', node_from_str)
        self.assertEqual(node_from_node.input, ['node_a'])

    def test_copy_op_attrs(self):
        """copy_op_attrs should only copy attrs supported by the target node"""
        # copy_op_attrs is used to transfer attrs from a fused op node
        # (e.g. _FusedConv2D) to a standalone op (e.g. Conv2D)
        # any additional attrs of the fused op need to be ignored
        fused_op_str = '{"name":"model/conv2d/BiasAdd",'\
            + '"op":"_FusedConv2D","input":["input",'\
            + '"model/conv2d/Conv2D/ReadVariableOp",'\
            + '"model/conv2d/BiasAdd/ReadVariableOp",'\
            + '"model/p_re_lu/Neg"],"device":"/device:CPU:0",' \
            + '"attr":{"dilations":{"list":{"i":["1","1","1","1"]}},'\
            + '"T":{"type":"DT_FLOAT"},"data_format":{"s":"TkhXQw=="},'\
            + '"strides":{"list":{"i":["1","1","1","1"]}},'\
            + '"use_cudnn_on_gpu":{"b":true},'\
            + '"explicit_paddings":{"list":{}},'\
            + '"num_args":{"i":"2"},"epsilon":{"f":0},'\
            + '"padding":{"s":"VkFMSUQ="},'\
            + '"fused_ops":{"list":{"s":["Qmlhc0FkZA==","UHJlbHU="]}}}}'
        fused_op = testutils.node_proto_from_json(fused_op_str)
        node = rewrite.make_op_node('Conv2D', fused_op.input[0:2])
        rewrite.copy_op_attrs(source=fused_op, target=node)

        op_def = rewrite.get_op_def(node.op)
        allowed = set(attr.name for attr in op_def.attr)
        forbidden = any(attr for attr in node.attr if attr not in allowed)

        self.assertFalse(forbidden)
        # randomply check for some of the expected attributes
        self.assertTrue('padding' in node.attr)
        self.assertTrue('strides' in node.attr)

    def test_update_graph_def_given_empty_args(self):
        """update_graph_def should copy input as-is given empty dicts"""
        graph_def = testutils.get_sample_graph_def()
        updated_graph = rewrite.update_graph_def(graph_def, {}, {})
        self.assertEqual(graph_def, updated_graph)

    def test_update_graph_def_given_removed_nodes(self):
        """update_graph_def should remove nodes mapped to empty lists
           or None
        """
        graph_def = testutils.get_sample_graph_def()
        # basically removes the Keras Conv2D-layer named 'conv3'
        remove_nodes = {
            'model/conv3/Conv2D': [],
            'model/conv3/BiasAdd': None,
            'model/conv3/Relu': []
        }
        updated_graph = rewrite.update_graph_def(graph_def, remove_nodes, {})
        self.assertNotEqual(graph_def, updated_graph)
        # nodes must be removed
        for node_name in remove_nodes.keys():
            self.assertIsNone(_get_node_by_name(updated_graph, node_name))

    def test_update_graph_def_given_replaced_nodes(self):
        """update_graph_def should replace nodes mapped to new sub-graph"""
        graph_def = testutils.get_sample_graph_def()
        # let's replace the conv1 activation with log-sigmoid
        relu = _get_node_by_name(graph_def, 'model/conv1/Relu')
        neg = rewrite.make_op_node('Neg', list(relu.input), 'model/conv1/Neg')
        exp = rewrite.make_op_node('Exp', neg, 'model/conv1/Exp')
        add = rewrite.make_op_node('Add', [exp, 'one'], 'model/conv1/Add')
        inv = rewrite.make_op_node('Inv', add, 'model/conv1/Inv')
        replace_nodes = {
            'model/conv1/Relu': [neg, exp, add, inv]
        }
        updated_graph = rewrite.update_graph_def(graph_def, replace_nodes, {})
        for node_name in replace_nodes.keys():
            self.assertIsNone(_get_node_by_name(updated_graph, node_name))
        for node in list(replace_nodes.values())[0]:
            self.assertIsNotNone(_get_node_by_name(updated_graph, node.name))

    def test_update_graph_def_given_remapped_input(self):
        """update_graph_def should remap inputs given a non-empty mapping"""
        graph_def = testutils.get_sample_graph_def()
        # basically removes the Keras Conv2D-layer named 'conv3'
        remove_nodes = {
            'model/conv3/Conv2D': [],
            'model/conv3/BiasAdd': None,
            'model/conv3/Relu': []
        }
        # this time we also re-route the inputs proplerly
        remap_inputs = {
            'model/conv3/Relu': 'model/conv2/Relu'
        }
        updated_graph = rewrite.update_graph_def(graph_def, remove_nodes,
                                                 remap_inputs)
        self.assertNotEqual(graph_def, updated_graph)
        # nodes must be removed
        for node_name in remove_nodes.keys():
            self.assertIsNone(_get_node_by_name(updated_graph, node_name))
        # inputs must be remapped
        for node in updated_graph.node:
            for key in remove_nodes.keys():
                self.assertNotIn(key, node.input)

    def test_get_input_node_map_given_valid_graph(self):
        """get_input_node_map should accept valid graphs"""
        graph_def = testutils.get_sample_graph_def()
        input_nodes = rewrite.get_input_node_map(graph_def)
        self.assertGreater(len(input_nodes), 1)
        # randomly verify the existence of nodes in the map
        self.assertIn('model/conv1/BiasAdd', input_nodes)
        self.assertIn('model/flatten/Reshape', input_nodes)
        self.assertIn('model/output/MatMul', input_nodes)

    def test_get_input_node_map_given_duplicates(self):
        """get_input_node_map should raise ValueError given duplicate names"""
        graph_def = testutils.get_sample_graph_def()
        relu = _get_node_by_name(graph_def, 'model/conv3/Relu')
        neg = rewrite.make_op_node('Neg', list(relu.input), name='kate')
        dup = rewrite.make_op_node('Exp', neg, name='model/conv3/BiasAdd')
        replace_nodes = {
            'model/conv3/Relu': [neg, dup],
        }
        updated_graph = rewrite.update_graph_def(graph_def, replace_nodes, {})
        self.assertRaises(ValueError, lambda: rewrite.get_input_node_map(
                              updated_graph))

    def test_replace_matching_nodes(self):
        # case 1: unchanged copy if no matches
        graph_def = testutils.get_sample_graph_def()

        def _is_prelu(node): return node.op == 'Prelu'
        def _remove_node(node, map, mods): return []
        updated_graph_def, modifiers = rewrite.replace_matching_nodes(
            graph_def, predicate=_is_prelu, transform=_remove_node
        )
        self.assertEqual(modifiers, {})
        self.assertEqual(updated_graph_def, graph_def)
        # case 2: replaces matching nodes and keeps graph valid
        name_of_node_to_replace = 'model/conv2/Relu'
        new_name_of_replaced_node = ''
        def _must_replace(node): return node.name == name_of_node_to_replace

        def _convert_to_log_sigmoid(node, input_map, modifiers):
            """replace Relu with logarithmic sigmoid 1/(1+exp(-x))"""
            def _get_name(suffix):
                return rewrite.generate_name_from(node.name, input_map,
                                                  f'logSigmoid/{suffix}')
            nonlocal new_name_of_replaced_node
            # -x
            neg = rewrite.make_op_node('Neg', list(node.input),
                                       name=_get_name('Neg'))
            # exp(-x)
            exp = rewrite.make_op_node('Exp', neg, name=_get_name('Exp'))
            # constant tensor holding "1"
            res = rewrite.make_const_node(np.array([1], dtype=np.float32),
                                          name=_get_name('Var/resource'))
            # variable holding "1"
            one = rewrite.make_op_node('Identity', res, _get_name('Var'))
            # 1+exp(-x)
            add = rewrite.make_op_node('Add', [one, exp], _get_name('Add'))
            # 1/(1+exp-x)
            inv = rewrite.make_op_node('Inv', add, _get_name('Inv'))
            new_name_of_replaced_node = inv.name    # remember the output name
            return [neg, exp, res, one, add, inv]

        updated_graph_def, modifiers = rewrite.replace_matching_nodes(
            graph_def,
            predicate=_must_replace,
            transform=_convert_to_log_sigmoid)

        # replaced node must have been removed
        updated_nodes = rewrite.get_input_node_map(updated_graph_def)
        self.assertNotIn(name_of_node_to_replace, updated_nodes)
        # replaced node must not be referenced
        for _, node in updated_nodes.items():
            # nodes with inputs only
            if node.op not in ('Const', 'Placeholder'):
                self.assertNotIn(name_of_node_to_replace, node.input)

        # referenced to replaced node must point to last node in replacement
        original_nodes = rewrite.get_input_node_map(graph_def)
        replaced_references = [
            node.name for node in original_nodes.values()
            if name_of_node_to_replace in node.input
        ]
        for node_name in replaced_references:
            node = updated_nodes[node_name]
            self.assertIn(new_name_of_replaced_node, node.input)

    def test_generate_name_from_given_unique_name(self):
        """generate_name_from should return base name if name is unique"""
        # case #1: no slashes in name, name is unique -> return as-is
        node_map = {}
        base_name = 'name_without_slahes'
        name = rewrite.generate_name_from(base_name, node_map)
        self.assertEqual(name, base_name)
        # case #2: no slashes in name, name in map, name + suffix unique
        node_map[base_name] = rewrite.make_op_node('Neg', 'x', name=base_name)
        suffix = 'uniq'
        name = rewrite.generate_name_from(base_name, node_map, suffix=suffix)
        self.assertEqual(name, f'{base_name}/{suffix}')
        # case #3: slashes in name, name with slashes in map -> return 1st part
        base_name = 'name/suffix'
        node_map[base_name] = rewrite.make_op_node('Neg', 'y', name=base_name)
        name = rewrite.generate_name_from(base_name, node_map)
        self.assertEqual(name, 'name')
        # case #4: slashes in name, name in map, name + suffix unique
        base_name = 'name/suffix/op'
        name = rewrite.generate_name_from(base_name, node_map, suffix='uniq')
        self.assertEqual(name, 'name/suffix/uniq')

    def test_generate_name_from_given_duplicate_name(self):
        """generate_name_from should return new name if base name is taken"""
        node_map = {}
        # case #1: name without slashes -> return name + count
        base_name = 'name_without_slashes'
        node_map[base_name] = rewrite.make_op_node('Neg', 'x', name='_')
        name = rewrite.generate_name_from(base_name, node_map)
        self.assertEqual(name, base_name+'_1')
        # case #2: count needs to increment
        node_map[name] = rewrite.make_op_node('Neg', 'y', name='_')
        name = rewrite.generate_name_from(base_name, node_map)
        self.assertEqual(name, base_name+'_2')
        # case #3: name + suffix -> return name + suffix + count
        suffix = 'suffix'
        node_map[base_name+'/'+suffix] = rewrite.make_op_node('Inv', 'x',
                                                              name='_')
        name = rewrite.generate_name_from(base_name, node_map, suffix=suffix)
        self.assertEqual(name, f'{base_name}/{suffix}_1')
        # case #4: count needs to increment
        node_map[name] = rewrite.make_op_node('Inv', 'y', name='_')
        name = rewrite.generate_name_from(base_name, node_map, suffix=suffix)
        self.assertEqual(name, f'{base_name}/{suffix}_2')
        # case #5: slashes in name, name in map -> return name + count
        base_name = 'name_without_slashes/suffix'
        name = rewrite.generate_name_from(base_name, node_map)
        self.assertEqual(name, 'name_without_slashes_2')
        # case #6: slashes in name, name + suffix in map -> name+suffix+count
        base_name = 'name_without_slashes/suffix'
        name = rewrite.generate_name_from(base_name, node_map, suffix=suffix)
        self.assertEqual(name, f'{base_name}_2')

    def test_is_fused_op(self):
        """is_fused_op should be true if op is fused with BiasAdd+Activation"""
        missing_activation = testutils.node_proto_from_json(
            '{"name":"model/output/BiasAdd","op":"_FusedMatMul",'
            '"input":["model/dense/BiasAdd",'
            '"model/output/MatMul/ReadVariableOp",'
            '"model/output/BiasAdd/ReadVariableOp"],"device":"/device:CPU:0",'
            '"attr":{"transpose_b":{"b":false},"T":{"type":"DT_FLOAT"},'
            '"num_args":{"i": "1"},"epsilon":{"f": 0},'
            '"fused_ops":{"list":{"s":["Qmlhc0FkZA=="]}},'
            '"transpose_a":{"b":false}}}')
        self.assertFalse(
            rewrite.is_fused_op(missing_activation, 'MatMul', b'Relu'))
        fused_matmul = testutils.node_proto_from_json(
            '{"name":"model/dense/BiasAdd","op":"_FusedMatMul",'
            '"input":["model/flatten/Reshape",'
            '"model/dense/MatMul/ReadVariableOp",'
            '"model/dense/BiasAdd/ReadVariableOp","model/p_re_lu_2/Neg"],'
            '"device":"/device:CPU:0","attr":{"transpose_b":{"b":false},'
            '"T":{"type":"DT_FLOAT"},"num_args":{"i":"2"},"epsilon":{"f":0},'
            '"fused_ops":{"list":{"s":["Qmlhc0FkZA==","UHJlbHU="]}},'
            '"transpose_a":{"b":false}}}')
        self.assertTrue(rewrite.is_fused_op(fused_matmul, 'MatMul', b'Prelu'))

    def test_validate_supported_ops_given_valid_graph(self):
        """validate_supported_ops should accept valid graph_def"""
        graph_def = testutils.get_sample_graph_def()
        rewrite.validate_supported_ops(graph_def)

    def test_validate_supported_ops_given_invalid_graph(self):
        """validate_supported_ops should raise ValueError for unsupported op"""
        # case 1: unsupported op node
        graph_def = rewrite.GraphDef()
        unsupported_op = testutils.node_proto_from_json(
            '{"name":"model/p_re_lu_1/Relu","op":"Prelu","input":'
            '["model/add/add","model/p_re_lu_1/Neg"]}')
        graph_def.node.extend([unsupported_op])
        self.assertRaises(ValueError,
                          lambda: rewrite.validate_supported_ops(graph_def))
        # case 2: unsupported fused op
        unsupported_fused_op = testutils.node_proto_from_json(
            '{"name":"model/dense/BiasAdd","op":"_FusedMatMul",'
            '"input":["model/flatten/Reshape",'
            '"model/dense/MatMul/ReadVariableOp",'
            '"model/dense/BiasAdd/ReadVariableOp","model/p_re_lu_2/Neg"],'
            '"device":"/device:CPU:0","attr":{"transpose_b":{"b":false},'
            '"T":{"type":"DT_FLOAT"},"num_args":{"i":"2"},"epsilon":{"f":0},'
            '"fused_ops":{"list":{"s":["Qmlhc0FkZA==","UHJlbHU="]}},'
            '"transpose_a":{"b":false}}}')
        graph_def = rewrite.GraphDef()
        graph_def.node.extend([unsupported_fused_op])
        self.assertRaises(ValueError,
                          lambda: rewrite.validate_supported_ops(graph_def))


if __name__ == '__main__':
    unittest.main()
