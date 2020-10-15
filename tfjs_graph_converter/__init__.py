import os

# make tensorflow stop spamming messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
# disable CUDA devices - we only want the CPU do work with data
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

# bring in all modules
from tfjs_graph_converter import api                    # noqa: F401,E402
from tfjs_graph_converter import common as constants    # noqa: F401,E402
from tfjs_graph_converter import converter              # noqa: F401,E402
from tfjs_graph_converter import graph_rewrite_util     # noqa: F401,E402
from tfjs_graph_converter import optimization           # noqa: F401,E402
from tfjs_graph_converter import util                   # noqa: F401,E402
from tfjs_graph_converter import version                # noqa: F401,E402

__version__ = version.VERSION
VERSION = version.VERSION
