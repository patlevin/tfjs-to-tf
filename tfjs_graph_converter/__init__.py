import os

# bring in all modules
from tfjs_graph_converter import api                    # noqa: F401
from tfjs_graph_converter import common as constants    # noqa: F401
from tfjs_graph_converter import converter              # noqa: F401
from tfjs_graph_converter import graph_rewrite_util     # noqa: F401
from tfjs_graph_converter import optimization           # noqa: F401
from tfjs_graph_converter import util                   # noqa: F401
from tfjs_graph_converter import version                # noqa: F401

# make tensorflow stop spamming messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

__version__ = version.VERSION
VERSION = version.VERSION
