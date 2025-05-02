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
#  # DECOMON tutorial #3
#
# _**Local Robustness to Adversarial Attacks for classification tasks**_
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
# After training a model, we want to make sure that the model will give the same output for any images "close" to the initial one, showing some robustness to perturbation.
#
# In this notebook, we start from a classifier built on MNIST dataset that given a hand-written digit as input will predict the digit. This will be the first part of the notebook.
#
# <img src="./data/Plot-of-a-Subset-of-Images-from-the-MNIST-Dataset.png" alt="examples of hand-written digit" width="600"/>
#
# In the second part of the notebook, we will investigate the robustness of this model to unstructured modification of the input space: adversarial attacks. For this kind of attacks, **we vary the magnitude of the perturbation of the initial image** and want to assess that despite this noise, the classifier's prediction remain unchanged.
#
# <img src="./data/illustration_adv_attacks.jpeg" alt="examples of perturbated images" width="600"/>
#
# What we will show is the use of decomon module to assess the robustness of the prediction towards noise.

# %% [markdown]
# - When running this notebook on Colab, we need to install *decomon* if on Colab.
# - If you run this notebook locally, do it inside the environment in which you [installed *decomon*](https://airbus.github.io/decomon/main/install.html).

# %%
# On Colab: install the library
on_colab = "google.colab" in str(get_ipython())
if on_colab:
    import sys  # noqa: avoid having this import removed by pycln

    # install dev version for dev doc, or release version for release doc
    # !{sys.executable} -m pip install -U pip
    # !{sys.executable} -m pip install git+https://github.com/airbus/decomon@main#egg=decomon
    # install desired backend (by default tensorflow)
    # !{sys.executable} -m pip install "tensorflow>=2.16"

# %% [markdown]
# ## Imports

# %matplotlib inline
# %%
import os
import os.path
import pickle as pkl
import time
from contextlib import closing

import ipywidgets as widgets
import keras
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interact
from keras.datasets import mnist
from keras.layers import Activation, Dense, Input
from keras.models import Sequential

# %% [markdown]
# ## Load images
#
# We load MNIST data from keras datasets.
#

# %%
img_rows, img_cols = 28, 28
(x_train, y_train_), (x_test, y_test_) = mnist.load_data()
x_train = x_train.reshape((-1, 784))
x_test = x_test.reshape((-1, 784))
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255.0
x_test /= 255.0
y_train = keras.utils.to_categorical(y_train_)
y_test = keras.utils.to_categorical(y_test_)

# %% [markdown]
# ## Learn the model (classifier for MNIST images)
#
# For the model, we use a small fully connected network. It is made of 6 layers with 100 units each and ReLU activation functions. **Decomon** is compatible with a large set of Keras layers, so do not hesitate to modify the architecture.
#

# %%
model = Sequential()
model.add(Dense(100, input_dim=784))
model.add(Activation("relu"))  # Decomon deduces tighter bound when splitting Dense and activation
model.add(Dense(100))
model.add(Activation("relu"))
model.add(Dense(10, activation="softmax"))

# %%
model.compile("adam", "categorical_crossentropy", metrics=["acc"])

model.fit(x_train, y_train, batch_size=32, shuffle=True, validation_split=0.2, epochs=3)

# %%
model.evaluate(x_test, y_test, batch_size=32)

# %% [markdown]
# After training, we see that the assessment of performance of the model on data that was not seen during training shows pretty good results: around 0.97 (maximum value is 1). It means that out of 100 images, the model was able to guess the correct digit for 97 images. But how can we guarantee that we will get this performance for images different from the ones in the test dataset?
#
# - If we perturbate a "little" an image that was well predicted, will the model stay correct?
# - Up to which perturbation?
# - Can we guarantee that the model will output the same digit for a given perturbation?
#
# This is where decomon comes in.
#
# <img src="./data/decomon.jpg" alt="Decomon!" width="400"/>
#
#

# %% [markdown]
# ## Applying decomon for local robustness to misclassification
#
# In this section, we detail how to prove local robustness to misclassification. Misclassification can be studied with the global optimisation of a function f:
#
# $$ f(x; \Omega) = \max_{z\in \Omega} \text{NN}_{j\not= i}(z) - \text{NN}_i(z)\;\; \text{s.t}\;\; i = argmax\;\text{NN}(x)$$
#
# If the maximum of f is **negative**, this means that whathever the input sample from the domain, the value outputs by the neural network NN for class i will always be greater than the value output for another class. Hence, there will be no misclassification possible. This is **adversarial robustness**.
#
# <img src="./data/tuto_3_formal_robustness.png" alt="Decomon!" width="700"/>
#
# In that order, we will use the [decomon](https://gheprivate.intra.corp/CRT-DataScience/decomon/tree/master/decomon) library. Decomon combines several optimization trick, including linear relaxation
# to get state-of-the-art outer approximation.
#
# To use **decomon** for **adversarial robustness** we first need the following imports:
# + *from decomon.models import clone*: to convert our current Keras model into another neural network nn_model. nn_model shares the same weights but outputs information that will be used to derive our formal bounds. For a sake of clarity, how to get such bounds is hidden to the user
#
# + *from decomon import get_adv_box*: a genereric method to get an upper bound of the funtion f described previously. If the returned value is negative, then we formally assess the robustness to misclassification.
#
# + *from decomon import check_adv_box*: a generic method that computes the maximum of a lower bound of f. Eventually if this value is positive, it demonstrates that the function f takes positive value. It results that a positive value formally proves the existence of misclassification.
#

# %%
from decomon import get_adv_box
from decomon.models import clone

# %% [markdown]
# For computational efficiency, we convert the model into its decomon version once and for all.
# Note that the decomon method will work on the non-converted model.

# %%
decomon_model = clone(model, method="crown")


# %% [markdown]
# We offer an interactive visualisation of the basic adversarial robustness method from decomon **get_adv_box**. We randomly choose 10 test images use **get_adv_box** to assess their robustness to misclassification pixel perturbations. The magnitude of the noise on each pixel is independent and bounded by the value of the variable epsilon. The user can reset the examples and vary the noise amplitude.
#
# Note one of the main advantage of decomon: **we can assess robustness on batches of data!**
#
# Circled in <span style="color:green">green</span> are examples that are formally assessed to be robust, <span style="color:orange">orange</span> examples that could be robust and  <span style="color:red">red</span> examples that are formally non robust


# %%
def frame(epsilon, reset=0, filename=".hidden_index.pkl"):
    n_cols = 5
    n_rows = 2
    n_samples = n_cols * n_rows
    if reset:
        index = np.random.permutation(len(x_test))[:n_samples]

        with closing(open(filename, "wb")) as f:
            pkl.dump(index, f)
        # save data
    else:
        # check that file exists

        if os.path.isfile(filename):
            with closing(open(filename, "rb")) as f:
                index = pkl.load(f)
        else:
            index = np.arange(n_samples)
            with closing(open(filename, "wb")) as f:
                pkl.dump(index, f)
    # x = np.concatenate([x_test[0:1]]*10, 0)
    x = x_test[index]

    x_min = np.maximum(x - epsilon, 0)
    x_max = np.minimum(x + epsilon, 1)

    n_cols = 5
    n_rows = 2
    fig, axs = plt.subplots(n_rows, n_cols)

    fig.set_figheight(n_rows * fig.get_figheight())
    fig.set_figwidth(n_cols * fig.get_figwidth())
    plt.subplots_adjust(hspace=0.2)  # increase vertical separation
    axs_seq = axs.ravel()

    source_label = y_test[index]

    start_time = time.process_time()

    upper = get_adv_box(decomon_model, x_min, x_max, source_labels=source_label, n_sub_boxes=1)
    # lower = check_adv_box(decomon_model, x_min, x_max, source_labels=source_label, n_sub_boxes=1)
    lower = 0.0 * upper - 1
    end_time = time.process_time()

    count = 0
    time.sleep(1)
    r_time = "{:.2f}".format(end_time - start_time)
    fig.suptitle(
        "Formal Robustness to Adversarial Examples with eps={} running in {} seconds".format(epsilon, r_time),
        fontsize=16,
    )
    for i in range(n_cols):
        for j in range(n_rows):
            ax = axs[j, i]
            ax.imshow(x[count].reshape((28, 28)), cmap="Greys")
            robust = "ROBUST"
            if lower[count] >= 0:
                if upper[count] <= 0:
                    import pdb

                    pdb.set_trace()
                color = "red"
                robust = "NON ROBUST"
            elif upper[count] < 0:
                color = "green"
            else:
                color = "orange"
                robust = "MAYBE ROBUST"

            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            # Create a Rectangle patch
            rect = patches.Rectangle((0, 0), 27, 27, linewidth=3, edgecolor=color, facecolor="none")
            ax.add_patch(rect)
            ax.set_title(robust)
            count += 1

    plt.show()


interact(
    frame,
    epsilon=widgets.FloatSlider(
        value=0.0,
        min=0.0,
        max=25.0 / 255.0,
        step=0.0001,
        continuous_update=False,
        readout_format=".4f",
    ),
    reset=widgets.IntSlider(value=0.0, min=0, max=1, step=1, continuous_update=False),
    fast=widgets.IntSlider(value=1.0, min=0, max=1, step=1, continuous_update=False),
)

# %% [markdown]
# As explained previously, the method **get_adv_box** output a constant upper bound that is valid on the whole domain.
# Sometimes, this bound can be too lose and needs to be refined by splitting the input domain into sub domains.
# Several heuristics are possible and you are free to develop your own or take an existing one of the shelf.

# %% [markdown]
# ## Improving the verification: bounding adversarial robustness

# %% [markdown]
# Unlike the verification studied in the previous tutorials 1 & 2, **adversarial robustness** relies on the largest difference taken by two outputs.
# Instead of bouding the output of the neural network, we want to bound the function that measures **adversarial robustness**, expressed hereafter:
# $$ f(x; \Omega) = \max_{z\in \Omega} \text{NN}_{j\not= i}(z) - \text{NN}_i(z)\;\; \text{s.t}\;\; i = argmax\;\text{NN}(x)$$
#
# In this section, we adapt the previous cells of the notebook to optimize formal methods for adversarial robustness. In that aim, we add another input to our formal method: a matrix **C** that encompasses the information. C is a triangular matrix as the following:
# $$
# C_{p,q} = \left\{
#     \begin{array}\\
#         1 & \mbox{if } \ p = \ q \not= \ i \\
#         0 & \mbox{if } \ p = \ q = \ i \\
#         -1 & \mbox{if } \ p \not= \ q  \mbox{ and } \ q = \ i \\
#         -1 & \mbox{else.}
#     \end{array}
# \right.
# $$
#
# With the next cells, we can observe by varying the radius of the perturbation that the new formal method outperforms the previous one and sometimes allows to demonstrate the **adversarial robustness**

# %%
C = Input((10, 10))
decomon_model_adv = clone(model, backward_bounds=[C])


# %%
def frame(eps=0.1, n_sub_boxes=1):
    # select 10 images
    indices = np.array([np.where(y_test_ == i)[0][0] for i in range(5)])
    images = x_test[indices]
    # we create a matrix that stores the information l_j - l_i
    C = np.diag([1.0] * 10)[None] - y_test[indices, :, None]

    adv_crown = get_adv_box(
        decomon_model, images - eps, images + eps, source_labels=y_test[indices], n_sub_boxes=n_sub_boxes
    )
    start_time = time.process_time()

    adv_better = get_adv_box(
        decomon_model_adv, images - eps, images + eps, source_labels=y_test[indices], n_sub_boxes=n_sub_boxes
    )

    end_time = time.process_time()

    n_rows = 2
    n_cols = 5

    fig, axs = plt.subplots(n_rows, n_cols)

    fig.set_figheight(n_rows * fig.get_figheight())
    fig.set_figwidth(n_cols * fig.get_figwidth())
    plt.subplots_adjust(hspace=0.2)  # increase vertical separation
    axs_seq = axs.ravel()

    count = 0
    time.sleep(1)
    r_time = "{:.2f}".format(end_time - start_time)
    fig.suptitle(
        "Formal Robustness to Adversarial Examples with eps={} (n_splits={}) running in {} seconds".format(
            eps, n_sub_boxes, r_time
        ),
        fontsize=16,
    )

    for i in range(n_rows):
        for j in range(n_cols):
            ax = axs[i, j]
            ax.imshow(images[j].reshape((28, 28)), cmap="Greys")

            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # Create a Rectangle patch
            if i == 0:
                if adv_crown[j] >= 0:
                    color = "orange"
                else:
                    color = "green"
            else:
                if adv_better[j] > 0:
                    color = "orange"
                else:
                    color = "green"
            rect = patches.Rectangle((0, 0), 27, 27, linewidth=3, edgecolor=color, facecolor="none")
            ax.add_patch(rect)

    plt.show()


interact(
    frame,
    eps=widgets.FloatSlider(
        value=0.0103,
        min=0.0,
        max=10.0 / 255.0,
        step=0.0001,
        continuous_update=False,
        readout_format=".4f",
    ),
    n_sub_boxes=widgets.IntSlider(value=1.0, min=1, max=20, step=1, continuous_update=False),
)

# %% [markdown]
# The output of the model will be 0 for the groundtruth label.
# If every other outputs have negative values, this implies that the prediction is robust to misclassification in the considered input region.
# The first rown contains the first formal method, while the second row is its optimized version. We observe that the optimized version may demonstrate robustness while the original formal method cannot.
