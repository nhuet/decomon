from __future__ import absolute_import
from tensorflow.keras.layers import Layer
from abc import ABC, abstractmethod

#  the different forward (from input to output) linear based relaxation perturbation analysis

# propgation of constant bounds from input to output
class F_IBP:
    name = "ibp"


# propagation of affine bounds from input to output
class F_FORWARD:
    name = "forward"


# propagation of constant and affines bounds from input to output
class F_HYBRID:
    name = "hybrid"


# create static variables for varying convex domain
class Ball:
    name = "ball"  # Lp Ball around an example


class Box:
    name = "box"  # Hypercube


class Vertex:
    name = "vertex"  # convex set represented by its vertices
    # (no verification is proceeded to assess that the set is convex)

class DEEL_LIP:
    name='deel-lip>'


class StaticVariables:
    """
    Storing static values on the number of input tensors for our layers
    """

    def __init__(self, dc_decomp=False, mode=F_HYBRID.name):
        """

        :param dc_decomp: boolean that indicates whether we return a difference of convex decomposition of our layer
        gradient
        :param mode: type of Forward propagation (IBP, Forward or Hybrid)
        """

        self.mode = mode

        if self.mode == F_HYBRID.name:
            nb_tensors = 7
        elif self.mode == F_IBP.name:
            nb_tensors = 2
        elif self.mode == F_FORWARD.name:
            nb_tensors = 5
        else:
            raise NotImplementedError("unknown forward mode {}".format(mode))

        if dc_decomp:
            nb_tensors += 2

        self.nb_tensors = nb_tensors


class DecomonLayer(ABC, Layer):
    """
    Abstract class that contains the common information of every implemented layers for Forward LiRPA
    """

    def __init__(
        self, convex_domain={}, dc_decomp=False, mode=F_HYBRID.name, finetune=False, shared=False, fast=True, **kwargs
    ):
        """

        :param convex_domain: type of convex input domain (None or dict)
        :param dc_decomp: boolean that indicates whether we return a
        difference of convex decomposition of our layer
        :param mode: type of Forward propagation (IBP, Forward or Hybrid)
        :param kwargs: extra parameters
        """
        super(DecomonLayer, self).__init__(**kwargs)

        self.nb_tensors = StaticVariables(dc_decomp, mode).nb_tensors
        self.dc_decomp = dc_decomp
        self.convex_domain = convex_domain
        self.mode = mode
        self.finetune = finetune  # extra optimization with hyperparameters
        self.frozen_weights = False
        self.frozen_alpha = False
        self.shared = shared
        self.fast = fast
        self.init_layer = False
        self.linear_layer=False
        self.has_backward_bounds = False # optimizing Forward LiRPA for adversarial perturbation

    def build(self, input_shape):
        """

        :param input_shape:
        :return:
        """
        pass

    @abstractmethod
    def call(self, inputs, **kwargs):
        """

        :param inputs:
        :return:
        """
        pass

    def set_linear(self, bool_init):
        self.linear_layer = bool_init

    def get_linear(self):
        return False

    @abstractmethod
    def compute_output_shape(self, input_shape):
        """

        :param input_shape:
        :return:
        """
        pass

    def reset_layer(self, layer):
        """

        :param layer:
        :return:
        """
        pass

    def join(self, bounds):
        """

        :param bounds:
        :return:
        """
        raise NotImplementedError()

    def freeze_weights(self):
        pass

    def unfreeze_weights(self):
        pass

    def freeze_alpha(self):
        pass

    def unfreeze_alpha(self):
        pass

    def reset_finetuning(self):
        pass

    def shared_weights(self, layer):
        pass

    def split_kwargs(self, **kwargs):
        # necessary for InputLayer
        # 'mode', 'dc_decomp', 'convex_domain', 'finetune', 'shared', 'fast'])
        pass

    def set_back_bounds(self, has_backward_bounds):
        pass
