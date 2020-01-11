# SPDX-License-Identifier: MIT
# Copyright © 2020 Patrick Levin
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import sys
import time

import tensorflow as tf
import tensorflowjs as tfjs
import tensorflowjs.converters.common as tfjs_common

import tfjs_graph_converter.api as api
import tfjs_graph_converter.common as common
import tfjs_graph_converter.version as version

def get_arg_parser():
    """
    Create the argument parser for the converter binary.
    """
    parser = argparse.ArgumentParser(description='TensorFlow.js Graph Model converter.')
    parser.add_argument(
        common.CLI_INPUT_PATH,
        nargs='?',
        type=str,
        help='Path to the TFJS Graph Model directory containing the model.json')
    parser.add_argument(
        common.CLI_OUTPUT_PATH,
        nargs='?',
        type=str,
        help='For output format "{}", a SavedModel target directory. '
        'For output format "{}", a frozen model file.'.format(
            common.CLI_SAVED_MODEL, common.CLI_FROZEN_MODEL)
    )
    parser.add_argument(
        '--' + common.CLI_OUTPUT_FORMAT,
        type=str,
        default=common.CLI_FROZEN_MODEL,
        choices=set([common.CLI_SAVED_MODEL, common.CLI_FROZEN_MODEL]),
        help='Output format. Default: {}.'.format(common.CLI_FROZEN_MODEL)
    )
    default_tag = tf.saved_model.SERVING
    parser.add_argument(
        '--' + common.CLI_SAVED_MODEL_TAGS,
        type=str,
        default=default_tag,
        help='Tags of the MetaGraphDef to save, in comma separated string '
        'format. Defaults to "{}". Applicable only if output format '
        'is {}'.format(default_tag, common.CLI_SAVED_MODEL)
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

def convert(arguments):
    """
    Convert a TensorflowJS-model to a TensorFlow-model.

    Args:
        arguments: List of command-line arguments   
    """
    args = get_arg_parser().parse_args(arguments)
    if args.show_version:
        print("\ntfjs_graph_converter {}\n".format(version.VERSION))
        print("Dependency versions:")
        print("    tensorflow {}".format(tf.version.VERSION))
        print("    tensorflowjs {}".format(tfjs.__version__))
        return

    def info(message, end=None):
        if not args.silence:
            print(message, end=end)

    if not args.input_path:
        raise ValueError(
            "Missing input_path argument. For usage, use the --help flag.")
    if not args.output_path:
        raise ValueError(
            "Missing output_path argument. For usage, use the --help flag.")

    info("TensorFlow.js Graph Model Converter\n")
    info("Graph model:    {}".format(args.input_path))
    info("Output:         {}".format(args.output_path))
    info("Target format:  {}".format(args.output_format))
    info("\nConverting....", end=" ")

    start_time = time.perf_counter()

    if args.output_format == common.CLI_FROZEN_MODEL:
        api.graph_model_to_frozen_graph(args.input_path, args.output_path)
    elif args.output_format == common.CLI_SAVED_MODEL:
        api.graph_model_to_saved_model(
            args.input_path, args.output_path, args.saved_model_tags)
    else:
        raise ValueError(
            "Unsupported output format: {}".format(args.output_format))

    end_time = time.perf_counter()
    info("Done.")
    info("Conversion took {0:.3f}s".format(end_time - start_time))

    return

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
        print(ex)

if __name__ == '__main__':
    tf.app.run(main=main, argv=[' '.join(sys.argv[1:])])