import os

# make tensorflow stop spamming messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

from tfjs_graph_converter import api
from tfjs_graph_converter import version
from tfjs_graph_converter import common
from tfjs_graph_converter import converter
from tfjs_graph_converter import util

__version__ = version.VERSION
VERSION = version.VERSION
