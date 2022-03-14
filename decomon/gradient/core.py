from __future__ import absolute_import
from tensorflow.keras.layers import Layer
from abc import ABC, abstractmethod
from decomon.layers.core import F_HYBRID


class GradientLayer(ABC, Layer):
    """
    Abstract class that contains the common information of every implemented layers
    """

    def __init__(
        self, convex_domain={}, mode=F_HYBRID.name, input_mode=F_HYBRID.name, finetune=False, shared=False, **kwargs
    ):
        self.convex_domain=convex_domain
        self.mode=mode
        self.input_mode = input_mode
