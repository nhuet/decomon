import json

import numpy as np
import tensorflow as tf

from tensorboard.plugins.graph.keras_util import keras_model_to_graph_def


class DoublingLayer(tf.keras.layers.Layer):
    """Layer doubling artificially the inputs."""

    def call(self, inputs):
        return inputs, inputs


# Model definition
input = tf.keras.Input(shape=(2,))
dbling_layer = DoublingLayer()
reducing_layer = tf.keras.layers.Add()
outputs = reducing_layer(dbling_layer(input))
model = tf.keras.Model(inputs=[input], outputs=outputs)

# Show the config
model_config_json = model.to_json(indent=2)
print("\n** json model config:\n")
print(model_config_json)

# Computation example with original model
print("** Computation with original model:")
print(model(np.array([1, 2])))

# Import the keras model from the json + computation  => valid json config
model2 = tf.keras.models.model_from_json(model_config_json, custom_objects={"DoublingLayer": DoublingLayer})
print("\n** Computation with model loaded from json:")
print(model(np.array([1, 2])))

# Try to import it with tensorboard graph plugin  => error with reducing node
print("\n** Try to convert json config to graph def\n")
print(keras_model_to_graph_def(json.loads(model_config_json)))
