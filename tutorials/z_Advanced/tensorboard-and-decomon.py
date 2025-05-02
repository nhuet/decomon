# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Tensorboard with decomon (not yet working with keras 3 and decomon>0.1.1)
#
#
# In this notebook, we show how to have a look to the graph of a decomon model.
#
# We use here the same model as in [tutorial 1](../tutorial1_sinus-interactive.ipynb) and you should refer to it for any details about how it works.
#

# %% [markdown]
# ## WARNING: Not working yet with keras 3!
#
# For now, tensorboard is not keras 3 ready, and thus this notebook does not work properly since decomon is now using [keras 3](https://keras.io/keras_3/).
# See [this issue](https://github.com/tensorflow/tensorboard/issues/6686) on tensorboard github.

# %% [markdown]
# ## Prerequisites: tensorboard (+ tensorflow + keras) >= 2.13
#
# Because decomon models have specificities, visualizing them with tensorboard reveal some bug in previous versions of the library. The bug is fixed starting from 2.13 so you need at least this version of tensorboard (and thus tensorflow and keras, for compatibility) to make this notebook work.
#
#
#

# %% [markdown]
# ### On Colab
#
# We need to ensure the version of tensorboard, and then we install decomon.
#

# %%
# On Colab: install the library
on_colab = "google.colab" in str(get_ipython())
if on_colab:
    import sys  # noqa: avoid having this import removed by pycln

    # install dev version for dev doc, or release version for release doc
    # !{sys.executable} -m pip install -U pip
    # !{sys.executable} -m pip install git+https://github.com/airbus/decomon@main#egg=decomon
    # install desired backend (by default tensorflow)
    # !{sys.executable} -m pip install "tensorflow>=2.16 tensorboard>=2.16"

# %% [markdown]
# ### On binder
#
# We prepared the proper environment already with tensorboard. However, **we were not successful in making the tensorboard visualization available on binder**. The magic command  `%tensorboard` seems not to work properly.
#
# We tried also to use [jupyter-server-proxy](https://github.com/jupyterhub/jupyter-server-proxy) as suggested in [this example](https://github.com/binder-examples/tensorboard). The tensorboard visualization should be then available at `{base_url_of_binder_runner}/proxy/6006`. However nothing is to be seen, even though the title of the tab is indeed set to "tensorboard".

# %% [markdown]
# ## Imports

# %%
# %load_ext tensorboard

from datetime import datetime

import keras
import numpy as np
import tensorboard
from keras.layers import Activation, Dense, Input
from keras.models import Sequential

from decomon.models import clone
from decomon.wrapper import get_lower_box, get_upper_box

print("Notebook run using keras:", keras.__version__)
print("Notebook run using tensorboard:", tensorboard.__version__)

# %% [markdown]
# ## Initial Keras model

# %% [markdown]
# ### Build model
#
# The sinusoide funtion is defined on a $[-1 ; 1 ]$ interval. We put a factor in the sinusoide to have several periods of oscillations.
#

# %%
x = np.linspace(-1, 1, 1000)
y = np.sin(10 * x)

# %% [markdown]
# We approximate this function by a fully connected network composed of 4 hidden layers of size 100, 100, 20 and 20 respectively. Rectified Linear Units (ReLU) are chosen as activation functions for all the neurons.

# %%
layers = []
layers.append(Input((1,)))
layers.append(Dense(100, activation="linear", name="dense1"))
layers.append(Activation("relu", name="relu1"))
layers.append(Dense(100, activation="linear", name="dense2"))
layers.append(Activation("relu", name="relu2"))
layers.append(Dense(20, activation="linear", name="dense3"))
layers.append(Activation("relu", name="relu3"))
layers.append(Dense(20, activation="linear", name="dense4"))
layers.append(Activation("relu", name="relu4"))
layers.append(Dense(1, activation="linear", name="dense5"))
model = Sequential(layers)

# %% [markdown]
# ### Fit model

# %% [markdown]
# Uncomment the cell below if you want to see the op graph generated during model fit, later in [tensorboard](#Tensorboard).

# %% [raw]
# model.compile("adam", "mse")
#
# # Define the Keras TensorBoard callback.
# logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
#
# model.fit(x, y, batch_size=32, shuffle=True, epochs=10, verbose=1, callbacks=[tensorboard_callback])
# # verbose=0 removes the printing along the training, *but* it prevent op graph in tensorboard (?!)
# # verbose=2 does not allow neither op graph in tensorboard, beware !

# %% [markdown]
# ## Conversion to decomon model

# %%
# convert our model into a decomon model:
decomon_model = clone(model, method="forward-ibp")  # method is optionnal

# %% [markdown]
# ## Visualization

# %% [markdown]
# ### Serialization to json

# %%
print(model.to_json())

# %%
print(decomon_model.to_json())

# %% [markdown]
# ### Visualization with Graphviz and pydot

# %% [markdown]
# You need to install pydot and graphviz to make it work. If available, uncomment the 2 next cells.

# %% [raw]
# tf.keras.utils.plot_model(
#     model,
#     to_file="model.png",
#     show_shapes=False,
#     show_dtype=False,
#     show_layer_names=True,
#     rankdir="TB",
#     expand_nested=False,
#     dpi=96,
#     layer_range=None,
#     show_layer_activations=False,
# )

# %% [raw]
# tf.keras.utils.plot_model(
#     decomon_model,
#     to_file="decomon_model.png",
#     show_shapes=False,
#     show_dtype=False,
#     show_layer_names=True,
#     rankdir="TB",
#     expand_nested=False,
#     dpi=96,
#     layer_range=None,
#     show_layer_activations=False,
# )

# %% [markdown]
# ### Tensorboard

# %% [markdown]
# We create a log file by setting respectively the Keras and Decomon models to a tensorboard callback.

# %%
logdir = "logs/keras-graph/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir, write_graph=True)
tensorboard_callback.set_model(model)

# %%
logdir = "logs/decomon-graph/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir, write_graph=True)
tensorboard_callback.set_model(decomon_model)

# %% [markdown]
# We launch tensorboard to visualize the graph.
# As the op graph is not available without fit, you need to select tag "keras" and graph type "Conceptual graph" on  the right to make it work.
# In "Run" drop-down menu, select "decomon-graph" and then double click on the big node to develop the graph.

# %%
# %tensorboard --logdir logs
