from keras.layers import AveragePooling2D

from decomon.layers import DecomonLayer


class DecomonAveragePooling2D(DecomonLayer):
    layer: AveragePooling2D
    increasing = True
