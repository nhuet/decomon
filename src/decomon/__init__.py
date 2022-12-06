"""Decomon
-------
The goal of Decomon is to provide a simple interface to the latest explanation
techniques for certified perturbation analysis
"""


import sys

from . import layers, models
from .metrics.loss import get_adv_loss, get_lower_loss, get_model, get_upper_loss
from .utils import get_AB as get_grid_params
from .utils import get_AB_finetune as get_grid_slope
from .wrapper import (
    check_adv_box,
    get_adv_box,
    get_adv_noise,
    get_lower_box,
    get_lower_noise,
    get_range_box,
    get_range_noise,
    get_upper_box,
    get_upper_noise,
    refine_box,
)
from .wrapper_with_tuning import get_lower_box_tuning, get_upper_box_tuning

if sys.version_info >= (3, 8):
    from importlib.metadata import (  # pylint: disable=no-name-in-module
        PackageNotFoundError,
        version,
    )
else:
    from importlib_metadata import PackageNotFoundError, version


try:
    __version__ = version("decomon")
except PackageNotFoundError:
    # package is not installed
    pass
