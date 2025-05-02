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
# # How to implement its own custom decomon layer?
#
# <decomonlinks>
# <p align="center">
#   <img src="data/decomon.jpg" alt="Decomon!" width="100">
# </p>
#
#
#
# - &#x1F4DA; <a href="https://airbus.github.io/decomon"> Documentation </a>
# - <a href="https://github.com/airbus/decomon"> Github </a>
# - <a href="https://airbus.github.io/decomon/main/tutorials.html "> Tutorials </a>
#
# _Author: [Melanie DUCOFFE](https://fr.linkedin.com/in/m%C3%A9lanie-ducoffe-bbb53165)_
# </decomonlinks>
#
# When using decomon on customized Keras layers (or not already implemented in decomon), one has to implement their decomon counterpart.
# The easiest way is to simply implement their constant and affine relaxation, as explained in this notebook.

# %% [markdown]
# ## Imports

# %%
# Imports
import keras
import keras.ops as K
import numpy as np
from keras.layers import Dense, Input, Layer
from keras.models import Sequential

# from decomon import get_lower_box, get_upper_box
from decomon import clone
from decomon.keras_utils import batch_multid_dot
from decomon.layers import DecomonLayer

# %% [markdown]
# ## Custom keras layer and keras model

# %% [markdown]
# We implement here 2 keras layers:
# - a linear layer: simply doubling its input
# - a non-linear layer: squaring its input


# %%
class Double(Layer):
    """Doubling layer."""

    def call(self, inputs):
        """Take double."""
        return inputs * 2


class Square(Layer):
    """Square layer."""

    def call(self, inputs):
        """Take square."""
        return inputs**2


# %%
model = Sequential([Input((2,)), Double(), Dense(10), Square()])

# %%
model.layers[2].input.shape


# %% [markdown]
# ## Decomon custom layer implementation
#
# We need to derive from `DecomonLayer` and implement some methods.

# %% [markdown]
# ### Linear layer
#
# For the linear layer, we only have to give the proper affine representation of the layer (with proper shape), once we have specified that it is a linear layer.
# As we have a linear (or affine) layer, this representation is independent of the batch and will be given as such.
#
# More precisely, we need to return weights and bias `w` and `b` such that
#
#     layer(x) = x * w + b
#
# where the multiplication is actually `keras.ops.tensordot` on all non-batch axes of `x` and the the first non-batch ones of `w`, same number as the number of non-batch axes in `x`.
#
# This can performed via `batch_multi_dot()` from decomon (same function will be used for non-linear case).
#
# In the generic case, with 1 output, the shapes are:
#
#     - x ~ (batchsize,) + layer.input.shape[1:]
#     - b ~ layer.output.shape[1:]
#     - w ~ layer.input.shape[1:] + layer.output.shape[1:]
#
# We can also use *diagonal representation*: this means that `w` is represented only by its diagonal. (This only possible if input and output are of same shape). The shapes are:
#
#     - x ~ (batchsize,) + layer.input.shape[1:]
#     - b ~ layer.output.shape[1:]
#     - w ~ layer.output.shape[1:]
#
#     - layer.input.shape[1:] == layer.output.shape[1:]


# %%
class DecomonDouble(DecomonLayer):
    linear = True  # specifying that this is a linear layer
    diagonal = True  # specifying `w` is represented by its diagonal

    def get_affine_representation(self):
        bias_shape = self.layer.input.shape[
            1:
        ]  # a decomon layer has always an attribute `layer` which is the corresponding keras layer for which it is the decomon counterpart.
        w = 2 * keras.ops.ones(bias_shape)
        b = keras.ops.zeros(bias_shape)
        return w, b


# %% [markdown]
# #### Verification
#
# Let us check the affine representation by comparing it with actual output

# %%
# Instantiate the keras layer and build it
layer = Double()
layer(Input((2,)))

# Instantiate the corresponding decomon layer
decomon_layer = DecomonDouble(layer=layer)

# Keras input/output
x = K.convert_to_tensor(np.random.random((3, 2)), dtype=keras.config.floatx())  # ensure using same precision as default
layer_output_np = K.convert_to_numpy(layer(x))

# Recompute with affine representation
w, b = decomon_layer.get_affine_representation()
missing_batchsize = (False, True)  # specify that `w` is missing batchsize (but not x)
diagonal = (False, True)  # specify that `w` is represented by its diagonal
recomputed_layer_output = batch_multid_dot(x, w, missing_batchsize=missing_batchsize, diagonal=diagonal) + b
recomputed_layer_output_np = K.convert_to_numpy(recomputed_layer_output)

# Compare
np.testing.assert_almost_equal(recomputed_layer_output_np, layer_output_np)

print("Perfect!")


# %% [markdown]
# ### Non-linear layer
#
# For the non-linear layer, we need to give the constant and affine relaxation of the layer (with proper shape)
#
# #### Constant relaxation
#
# Given lower and upper bounds on layer input, we give lower and upper constant bounds on layer output (with a batchsize).
#
# #### Affine relaxation
# .
# Given lower and upper bounds on layer input, we need to return weights and biases `w_l`, `b_l`, `w_u`, and `b_u` such that
#
#     x * w_l + b_l <= layer(x) <= x * w_u + b_u
#
# where the multiplication is batch-wise, and on multiple axes (the first non-batch ones of `w`, same number as the number of non-batch axes in `x`). This is performed via `batch_multi_dot()` from decomon.
#
# In the generic case, with 1 output, the shape are:
#
#     - x ~ (batchsize,) + layer.input.shape[1:]
#     - b_l, b_l ~ (batchsize,) + layer.output.shape[1:]
#     - w_l, w_u ~ (batchsize,) + layer.input.shape[1:] + layer.output.shape[1:]
#
#
# We can also use *diagonal representation*: as before, this means that `w_l` and `w_u` will  be represented by their diagonal, so of the same shape as  `b_l` and `b_u`. Only possible if input and output of the layer share the same shape.
#
#     - x ~ (batchsize,) + layer.input.shape[1:]
#     - b_l, b_l ~ (batchsize,) + layer.output.shape[1:]
#     - w_l, w_u ~ (batchsize,) + layer.output.shape[1:]
#
#     - layer.input.shape[1:] == layer.output.shape[1:]
#
#


# %%
class DecomonSquare(DecomonLayer):
    diagonal = True  # specifying `w_l` and `w_u` are represented by their diagonal

    def forward_ibp_propagate(self, lower, upper):
        # image of bounds
        f_lower = lower**2
        f_upper = upper**2

        # lower bound: if same sign, the minimum between 2 images, if opposite signs, 0.
        lower_out = K.where(K.sign(lower * upper) > 0, K.minimum(f_lower, f_upper), 0.0)
        # upper_bound: the maximum between 2 images
        upper_out = K.maximum(f_lower, f_upper)

        return lower_out, upper_out

    def get_affine_bounds(self, lower, upper):
        # image of bounds
        f_lower = lower**2
        f_upper = upper**2

        # lower bound:
        # - opposite signs: 0 hyperplan
        # - same signs: tangent hyperplan at minimum point
        w_l = K.where(
            K.sign(lower * upper) > 0,
            K.where(
                lower < 0,
                2 * upper,
                2 * lower,
            ),
            0.0,
        )
        b_l = K.where(
            K.sign(lower * upper) > 0,
            K.where(
                lower < 0,
                -(upper**2),
                -(lower**2),
            ),
            0.0,
        )

        # upper bound: by convexity, hyperplan between lower and upper
        w_u = (f_upper - f_lower) / K.maximum(
            K.cast(keras.config.epsilon(), dtype=upper.dtype), upper - lower
        )  # avoid dividing by 0. -> replace by epsilon.
        b_u = f_lower - w_u * lower

        return w_l, b_l, w_u, b_u


# %% [markdown]
# #### Verification
#
# Let us check the relaxations

# %%
# Instantiate the keras layer and build it
layer = Square()
layer(Input((2,)))

# Instantiate the corresponding decomon layer
decomon_layer = DecomonSquare(layer=layer)

# Keras bounds/input/output
lowers = [-2, -1, 1]
uppers = [-1, 1, 2]

x_np = np.concatenate([np.random.random((1, 2)) * (u - l) + l for l, u in zip(lowers, uppers)], axis=0)
lower_np = np.concatenate([np.reshape([l, l], (1, 2)) for l in lowers], axis=0)
upper_np = np.concatenate([np.reshape([u, u], (1, 2)) for u in uppers], axis=0)

x = K.convert_to_tensor(x_np, dtype=keras.config.floatx())  # ensure using same precision as default
lower = K.convert_to_tensor(lower_np, dtype=keras.config.floatx())
upper = K.convert_to_tensor(upper_np, dtype=keras.config.floatx())

layer_output_np = K.convert_to_numpy(layer(x))


# constant bounds
lower_ibp, upper_ibp = decomon_layer.forward_ibp_propagate(lower, upper)

# affine bounds => computed constant bounds
w_l, b_l, w_u, b_u = decomon_layer.get_affine_bounds(lower, upper)
diagonal = (False, True)  # specify that `w` is represented by its diagonal
lower_affine = batch_multid_dot(x, w_l, diagonal=diagonal) + b_l
upper_affine = batch_multid_dot(x, w_u, diagonal=diagonal) + b_u


lower_affine_np = K.convert_to_numpy(lower_affine)
upper_affine_np = K.convert_to_numpy(upper_affine)
lower_ibp_np = K.convert_to_numpy(lower_ibp)
upper_ibp_np = K.convert_to_numpy(upper_ibp)


# comparison
assert (lower_affine_np <= layer_output_np).all()
assert (upper_affine_np >= layer_output_np).all()
assert (lower_ibp_np <= layer_output_np).all()
assert (upper_ibp_np >= layer_output_np).all()

print("Perfect!")

# %% [markdown]
# ## Convert to decomon model
#
# We need to specify the mapping between keras and decomon custom layers.

# %%
decomon_model = clone(
    model,
    mapping_keras2decomon_classes={Square: DecomonSquare, Double: DecomonDouble},  # custom layers mapping
    final_ibp=True,  # keep final constant bounds
    final_affine=False,  # drop final affine bounds
)

decomon_model.summary()

# %% [markdown]
# Get formal lower and upper bounds on a box domain [0,1] for inputs:

# %%
# Create a fake box with the right shape
x_min = np.zeros((1, 2))
x_max = np.ones((1, 2))
x_box = np.concatenate([x_min[:, None], x_max[:, None]], axis=1)

# Get lower and upper bounds
lower_bound, upper_bound = decomon_model.predict_on_single_batch_np(
    x_box
)  # more efficient than predict on very small batch

print(f"lower bound: {lower_bound}")
print(f"upper bound: {upper_bound}")

# %% [markdown]
# Compare with empirical bounds
#

# %%
keras_input = K.convert_to_tensor(np.random.random((100, 2)))
keras_output = K.convert_to_numpy(model(keras_input))
lower_empirical = np.min(keras_output, axis=0)
upper_empirical = np.max(keras_output, axis=0)

print(f"empirical lower bound: {lower_empirical}")
print(f"empirical upper bound: {upper_empirical}")

# %% [markdown]
# We should have (and the tightest, the best the approximation):
#
#     lower_bounds <= lower_empirical  , upper_empirical <= upper_bound
#

# %% [markdown]
# That's all folks!
