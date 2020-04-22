# SPDX-License-Identifier: MIT
# Copyright © 2020 Patrick Levin
"""Functions to fix various known issues with exported TFJS models"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import base64
from typing import Any, Dict, List, Optional

import tfjs_graph_converter.common as common


def _find_if_has_key(obj: Dict[str, Any], key: str,
                     of_type: Optional[type] = None) -> List[Any]:
    """
    Recursively find all objects with a given key in a dictionary

    Args:
        obj: Dictionary to search
        key: Key to find
        of_type: [optional] Type of the referenced item

    Returns:
        List of all objects that contain an item with the given key
        and matching type
    """
    def get_children(item: Any) -> List[Any]:
        return [val for val in item.values() if isinstance(val, dict)]
    found = []
    stack = get_children(obj)
    while len(stack) > 0:
        item = stack.pop()
        if key in item and (of_type is None or isinstance(item[key], of_type)):
            found.append(item)
        stack.extend(get_children(item))
    return found


def _convert_string_attrs(node: Dict[str, Any]) -> None:
    """
    Deep search string attributes (labelled "s" in GraphDef proto)
    and convert ascii code lists to base64-encoded strings if necessary
    """
    attr_key = common.TFJS_NODE_ATTR_KEY
    str_key = common.TFJS_ATTR_STRING_VALUE_KEY
    # some layers (e.g. PReLU) don't contain the `attr` key,
    # so test for its presence
    attrs = {}
    if attr_key in node:
        attrs = _find_if_has_key(node[attr_key], key=str_key, of_type=list)
    for attr in attrs:
        array = attr[str_key]
        # check if conversion is actually necessary
        if (len(array) > 0) and isinstance(array, list) \
                and isinstance(array[0], int):
            string = ''.join(map(chr, array))
            binary = string.encode('utf8')
            attr[str_key] = base64.encodebytes(binary)
        elif len(array) == 0:
            attr[str_key] = None


def _fix_dilation_attrs(node: Dict[str, Any]) -> None:
    """
    Search dilations-attribute and convert
    misaligned dilation rates if necessary see
    https://github.com/patlevin/tfjs-to-tf/issues/1
    """
    path = ['attr', 'dilations', 'list']
    values = node
    for key in path:
        if key in values:
            values = values[key]
        else:
            values = None
            break
    # if dilations are present, they're stored in 'values' now
    ints = common.TFJS_ATTR_INT_VALUE_KEY
    if values is not None and ints in values \
            and isinstance(values[ints], list):
        value = values[ints]
        if len(value) != 4:
            # must be NCHW-formatted 4D tensor or else TF can't handle it
            raise ValueError("Unsupported 'dilations'-attribute in node "
                             f'{node[common.TFJS_NAME_KEY]}')
        # check for [>1,>1,1,1], which is likely a mistranslated [1,>1,>1,1]
        if int(value[0], 10) > 1:
            values[ints] = ['1', value[0], value[1], '1']


def fix_node_attributes(message_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fix various known issues found "in the wild":
    • Node attributes in deserialised JSON may contain strings as lists of
      ascii codes when the TF GraphDef proto expects base64 encoded strings
    • 'dilation' attributes may be misaligned in a way unsupported by TF
    Further fixes will be added as issues are reported.

    Args:
        message_dict: Graph model formatted as parsed JSON dictionary

    Returns:
        Updated message dictionary with fixes applied if necessary
    """
    if common.TFJS_NODE_KEY in message_dict:
        nodes = message_dict[common.TFJS_NODE_KEY]
        for node in nodes:
            _convert_string_attrs(node)
            _fix_dilation_attrs(node)
    return message_dict
