# SPDX-License-Identifier: MIT
# Copyright © 2020 Patrick Levin
# ==============================================================================
"""High-level converter functions and CLI entry point"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import re
import sys
import time

import tensorflow as tf
import tensorflowjs as tfjs

import tfjs_graph_converter.api as api
import tfjs_graph_converter.common as common
import tfjs_graph_converter.version as version


class SplitCommaSeparatedValues(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values.split(','))


class SplitCommaSeparatedTuples(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        ok = re.search('^[^:,]+:[^:,]+(,[^:,]+:[^:,]+)*$', values)
        if not ok:
            raise argparse.ArgumentError(self,
                                         'Argument must be comma-separated '
                                         'pairs of name:key')
        pairs = values.split(',')
        setattr(namespace, self.dest, [tuple(p.split(':')) for p in pairs])


def get_arg_parser():
    """Create the argument parser for the converter binary."""
    parser = argparse.ArgumentParser(
        description='TensorFlow.js Graph Model converter.')
    parser.add_argument(
        common.CLI_INPUT_PATH,
        nargs='?',
        type=str,
        help='Path to the TFJS Graph Model directory containing the '
             'model.json')
    parser.add_argument(
        common.CLI_OUTPUT_PATH,
        nargs='?',
        type=str,
        help=f'For output format "{common.CLI_SAVED_MODEL}", '
        'a SavedModel target directory. '
        f'For output format "{common.CLI_FROZEN_MODEL}", '
        'a frozen model file.'
    )
    parser.add_argument(
        '--' + common.CLI_OUTPUT_FORMAT,
        type=str,
        default=common.CLI_FROZEN_MODEL,
        choices=set([common.CLI_SAVED_MODEL, common.CLI_FROZEN_MODEL]),
        help=f'Output format. Default: "{common.CLI_FROZEN_MODEL}".'
    )
    group = parser.add_argument_group(f'{common.CLI_SAVED_MODEL} specific',
                                      'Arguments that apply to SavedModel '
                                      'export only.')
    group.add_argument(
        '--' + common.CLI_SAVED_MODEL_TAGS,
        action=SplitCommaSeparatedValues,
        type=str,
        help='Tags of the MetaGraphDef to save, in comma separated string '
             f'format. Defaults to "{tf.saved_model.SERVING}".',
        metavar='TAG[,TAG1[,...]]'
    )
    group.add_argument(
        '--' + common.CLI_OUTPUTS,
        action=SplitCommaSeparatedValues,
        type=str,
        help='Outputs of the model to add to the signature separated by comma',
        metavar='OUTPUT[,OUTPUT1[,...]]'
    )
    group.add_argument(
        '--' + common.CLI_SIGNATURE_KEY,
        type=str,
        help='Specifies the signature key to be used in the MetaGraphDef. '
             f' REQUIRES "--{common.CLI_OUTPUTS}" to be set if specified. '
    )
    group.add_argument(
        '--' + common.CLI_METHOD_NAME,
        type=str,
        help='Specifies the signature method name used in the MetaGraphDef. '
             f' REQUIRES "--{common.CLI_OUTPUTS}" to be set if specified. '
    )
    group.add_argument(
        '--' + common.CLI_RENAME,
        action=SplitCommaSeparatedTuples,
        type=str,
        help='Specifies keys for inputs and outputs in the model signature. '
             'Mappings are specified using name:key with multiple values '
             'separated by comma.'
             f' REQUIRES "--{common.CLI_OUTPUTS}" to be set if specified. ',
        metavar='NAME:KEY[,NAME1:KEY1[,...]]'
    )
    parser.add_argument(
        '--' + common.CLI_VERSION,
        '-v',
        dest='show_version',
        action='store_true',
        help='Show versions of the converter and its dependencies'
    )
    parser.add_argument(
        '--' + common.CLI_SILENT_MODE,
        '-s',
        dest='silence',
        action='store_true',
        help='Suppress any output besides error messages'
    )
    return parser


def _get_signature(namespace: argparse.Namespace) -> dict:
    return {namespace.signature_key: {
        api.SIGNATURE_OUTPUTS: namespace.outputs,
        api.SIGNATURE_METHOD: namespace.method_name
    }} if namespace.outputs is not None else None


def _get_signature_keys(namespace: argparse.Namespace) -> api.RenameMap:
    if namespace.rename is not None:
        return api.RenameMap(namespace.rename)
    else:
        return None


def convert(arguments):
    """
    Convert a TensorflowJS-model to a TensorFlow-model.

    Args:
        arguments: List of command-line arguments
    """
    args = get_arg_parser().parse_args(arguments)
    if args.show_version:
        print(f"\ntfjs_graph_converter {version.VERSION}\n")
        print("Dependency versions:")
        print(f"    tensorflow {tf.version.VERSION}")
        print(f"    tensorflowjs {tfjs.__version__}")
        return

    def info(message, end=None):
        if not args.silence:
            print(message, end=end, flush=True)

    if not args.input_path:
        raise ValueError(
            "Missing input_path argument. For usage, use the --help flag.")
    if not args.output_path:
        raise ValueError(
            "Missing output_path argument. For usage, use the --help flag.")
    if args.output_format == common.CLI_SAVED_MODEL:
        if args.signature_key is not None and args.outputs is None:
            raise ValueError(f'--{common.CLI_SIGNATURE_KEY} requires '
                             f'--{common.CLI_OUTPUTS} to be specified')
        if args.method_name is not None and args.outputs is None:
            raise ValueError(f'--{common.CLI_METHOD_NAME} requires '
                             f'--{common.CLI_OUTPUTS} to be specified')
        if args.rename is not None and args.outputs is None:
            raise ValueError(f'--{common.CLI_RENAME} requires '
                             f'--{common.CLI_OUTPUTS} to be specified')

    info("TensorFlow.js Graph Model Converter\n")
    info(f"Graph model:    {args.input_path}")
    info(f"Output:         {args.output_path}")
    info(f"Target format:  {args.output_format}")
    info("\nConverting....", end=" ")

    start_time = time.perf_counter()

    if args.output_format == common.CLI_FROZEN_MODEL:
        api.graph_model_to_frozen_graph(args.input_path, args.output_path)
    elif args.output_format == common.CLI_SAVED_MODEL:
        api.graph_model_to_saved_model(
            args.input_path, args.output_path,
            tags=args.saved_model_tags,
            signature_def_map=_get_signature(args),
            signature_key_map=_get_signature_keys(args))
    else:
        raise ValueError(f"Unsupported output format: {args.output_format}")

    end_time = time.perf_counter()
    info("Done.")
    info(f"Conversion took {end_time-start_time:.3f}s")


def pip_main():
    """Entry point for pip-packaged binary

    Required because the pip-packaged binary calls the entry method
    without arguments
    """
    main([' '.join(sys.argv[1:])])


def main(argv):
    """
    Entry point for debugging and running the script directly

    Args:
        argv: Command-line arguments as a single, space-separated string
    """
    try:
        convert(argv[0].split(' '))
    except ValueError as ex:
        msg = ex.args[0] if len(ex.args) > 0 else ex
        print(f'Error: {msg}')


if __name__ == '__main__':
    tf.compat.v1.app.run(main=main, argv=[' '.join(sys.argv[1:])])
