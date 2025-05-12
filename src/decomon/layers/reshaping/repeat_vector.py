from keras.layers import RepeatVector

from decomon.layers import DecomonLayer


class DecomonRepeatVector(DecomonLayer):
    layer: RepeatVector
    increasing = True
