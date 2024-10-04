from keras.layers import ZeroPadding1D

from decomon.layers import DecomonLayer
from decomon.layers.utils import get_affine_representation_wo_bias
from decomon.types import Tensor


class DecomonZeroPadding1D(DecomonLayer):
    layer: ZeroPadding1D
    linear = True

    def get_affine_representation(self) -> tuple[Tensor, Tensor]:
        return get_affine_representation_wo_bias(self.layer, diagonal=self.diagonal)
