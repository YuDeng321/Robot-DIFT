"""
Set of global variables shared across robomimic.
"""
import os
# Sets debugging mode. Should be set at top-level script so that internal
# debugging functionalities are made active
DEBUG = False

# Whether to visualize the before & after of an observation randomizer
VISUALIZE_RANDOMIZER = False

# wandb entity, supplied by the runtime environment when needed
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "")

# wandb api key, supplied by the runtime environment when needed
WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")

try:
    from robomimic.macros_private import *
except ImportError:
    from robomimic.utils.log_utils import log_warning
    import robomimic
    log_warning(
        "No private macro file found!"\
        "\nIt is recommended to use a private macro file"\
        "\nTo setup, run: python {}/scripts/setup_macros.py".format(robomimic.__path__[0])
    )
