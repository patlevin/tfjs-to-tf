# SPDX-License-Identifier: MIT
# Copyright © 2020 Patrick Levin
# ==============================================================================
"""Commonly used constants and -types"""

from enum import IntEnum
from typing import Union


# Keys in the model.json file
TFJS_NODE_KEY = 'node'

TFJS_NODE_ATTR_KEY = 'attr'
TFJS_NODE_CONST_KEY = 'Const'
TFJS_NODE_NOOP_KEY = 'NoOp'
TFJS_NODE_PLACEHOLDER_KEY = 'Placeholder'

TFJS_ATTR_DTYPE_KEY = 'dtype'
TFJS_ATTR_SHAPE_KEY = 'shape'
TFJS_ATTR_VALUE_KEY = 'value'
TFJS_ATTR_STRING_VALUE_KEY = 's'
TFJS_ATTR_INT_VALUE_KEY = 'i'

TFJS_NAME_KEY = 'name'
TFJS_DATA_KEY = 'data'
TFJS_OP_KEY = 'op'

# CLI arguments
CLI_INPUT_PATH = 'input_path'
CLI_OUTPUT_PATH = 'output_path'
CLI_OUTPUT_FORMAT = 'output_format'
CLI_SAVED_MODEL_TAGS = 'saved_model_tags'
CLI_VERSION = 'version'
CLI_SAVED_MODEL = 'tf_saved_model'
CLI_FROZEN_MODEL = 'tf_frozen_model'
CLI_SILENT_MODE = 'silent'
CLI_OPTIMIZATION_TARGET = 'optimization_target'
CLI_OPTIMIZATION_NONE = 'none'
CLI_OPTIMIZATION_CPU = 'cpu'
CLI_OPTIMIZATION_GPU = 'gpu'
CLI_OUTPUTS = 'outputs'
CLI_SIGNATURE_KEY = 'signature_key'
CLI_METHOD_NAME = 'method_name'
CLI_RENAME = 'rename'
CLI_COMPATMODE = 'compat_mode'


class CompatMode(IntEnum):
    """Compatibility modes for converting models:

        NONE: use full optimisation and all TF ops
        TFJS: use harmonised data types for TFJS <2.4.x compatibility
        TFLITE: output model using TFLite builtins only (i.e. no fused ops)
    """
    NONE = 0
    TFJS = 1
    TFLITE = 2

    # argparse compatibility functions
    def __str__(self) -> str:
        return self.name.lower()

    def __repr__(self) -> str:
        return str(self)

    @staticmethod
    def argparse(s: str) -> Union[str, 'CompatMode']:
        try:
            return CompatMode[s.upper()]
        except KeyError:
            return s
