import os

from tfjs_graph_converter import version

# make tensorflow stop spamming messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

__version__ = version.VERSION
VERSION = version.VERSION
