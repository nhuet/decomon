from keras.layers import Permute

from decomon.layers import DecomonLayer


class DecomonPermute(DecomonLayer):
    layer: Permute
    increasing = True
