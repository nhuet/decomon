from keras.layers import ZeroPadding3D

from decomon.layers import DecomonLayer


class DecomonZeroPadding3D(DecomonLayer):
    layer: ZeroPadding3D
    increasing = True
