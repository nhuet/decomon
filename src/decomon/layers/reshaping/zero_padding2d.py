import keras.ops as K
from keras.layers import ZeroPadding2D

from decomon.layers import DecomonLayer


class DecomonZeroPadding2D(DecomonLayer):
    layer: ZeroPadding2D
    linear = True

    def get_affine_representation(self):
        input_shape = list(self.layer.input.shape[1:])
        N = K.prod(input_shape)
        W = K.eye(N)

        # step 1:
        W = W.reshape([N] + input_shape)
        W = self.layer(W)
        W_shape_output = list(W.shape[1:])
        W = W.reshape(input_shape + W_shape_output)
        b = K.zeros(self.layer.output.shape[1:])

        return W, b

    def forward_ibp_propagate(self, lower, upper):
        return self.layer(lower), self.layer(upper)
