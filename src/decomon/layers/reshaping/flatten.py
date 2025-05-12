from keras.layers import Flatten

from decomon.layers import DecomonLayer


class DecomonFlatten(DecomonLayer):
    layer: Flatten
    increasing = True
