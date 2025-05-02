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
#  # DECOMON tutorial #2
#
# _**Local Robustness to sensor noise for Regression**_
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
# Embedding simulation models developed during the design
# of a platform opens a lot of potential new functionalities
# but requires additional certification. Usually, these models require too much computing power, take too much time to run
# so we need to build an approximation of these models that can
# be compatible with operational constraints, hardware constraints, and real-time constraints. Also, we need to prove that
# the decisions made by the system using the surrogate model
# instead of the reference one will be safe.
#
# A first assessment that can be performed is the **robustness of the prediction given sensor noise**: demonstrating that despite sensor noise, the neural network prediction remains consistent.
#
# Local Robustness to **sensoir noise** can be performed efficiently thanks to formal robustness. In this notebook, we demonstrate how to derive deterministic upper and lower bounds of the output prediction of a neural network in the vicinity of a test sample.

# %% [markdown]
# This tutorial need decomon to be installed as well as [pandas](https://pandas.pydata.org/).
#
# - When running this notebook on Colab, we need to install *decomon* if on Colab.
# - If you run this notebook locally, do it inside the environment in which you [installed *decomon*](https://airbus.github.io/decomon/main/install.html) and [pandas](https://pandas.pydata.org/docs/getting_started/install.html).

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
# ## Toy Example: Electric Motor Temperature
#
# We will demonstrate how to perform **Local Robustness to sensoir noise** on a surrogate toy case.
# A neural network is trained to infer the temperature of a permanent-magnet synchronous motor ([PMSM](https://en.wikipedia.org/wiki/Synchronous_motor#Permanent-magnet_motors) ) given correlated features:
#
# + ambiant: Ambient temperature as measured by a thermal sensor located closely to the stator.
# + coolant: Coolant temperature. The motor is water cooled. Measurement is taken at outflow.
# + u_d: Voltage d-component
# + u_q: Voltage q-component
# + motor_speed
# + torque: Torque induced by current.
# + i_d: Current d-component
# + i_q: Current q-component
#
#
# The recorded temperature refers to the Permanent Magnet surface temperature (pm) representing the rotor temperature. This was measured with an infrared with 140 hrs recordings. Distinctive sessions are identified with "profile_id". You will find additional information in the [official data repository](https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature)

# %% [markdown]
# ### Download the dataset locally
#
# To download the data you need a [Kaggle](https://www.kaggle.com) account. Then you can download the dataset by clicking on the "download" button on the [official data repository](https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature). Unzip the file in the same directory as this notebook.
#
#
# You can also use the method described for Binder and Colab below.

# %% [markdown]
# ### Download the dataset on Binder or Colab
#
# If you run this notebook on Binderhub or Colab, follow this process to get the dataset:
#
# - Create a [Kaggle]() account.
# - Download a Kaggle API token by clicking on "Create New API Token" on your account page. You will get a kaggle.json file with the needed credentials.
# - Upload this kaggle.jon file on Binderhub or Colab. (You need to click on the directory icon on the left, and then on the upload button.)
# - Then run the next cell which will
#     - put the token at the right place with the right accesses,
#     - use the kaggle api to download the dataset,
#     - unzip it.
#

# %%
import socket

on_colab = "google.colab" in str(get_ipython())  # running on colab?
on_binder = socket.gethostname().startswith("jupyter-")  # running on binder? (not 100% sure but rather robust)

if on_colab or on_binder:
    # First of all, upload your kaggle api token kaggle.json
    # ! mkdir ~/.kaggle
    # ! mv kaggle.json ~/.kaggle
    # ! chmod 600 ~/.kaggle/kaggle.json
    # ! kaggle datasets download -d wkirgsn/electric-motor-temperature
    # ! unzip -o electric-motor-temperature.zip
    ...

# %% [markdown]
# ## Preprocessing: prepare the data and the neural network

# %%
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Activation, Dense
from keras.models import Sequential
from numpy.testing import assert_almost_equal

# %% [markdown]
# For the sake of context, we display statistical informations of this dataset.

# %%
data = pd.read_csv("measures_v2.csv")
data.describe()

# %% [markdown]
# Now it's time to dinstinguish our target (output, *pm*) and our features (inputs). We build the train and test set with a 80/20 ratio
# given the *profile_id*. Indeed we don't want to be biased by the recording session.

# %%
y = data["pm"]  # column "pm" is our target
X = data.drop(["pm"], axis=1)  # the other columns are our features

# sort given profile_id and split into train and test (80% of the sessions will be used for training the NN)
index = []
for i in range(X["profile_id"].min(), X["profile_id"].max()):
    if i in X["profile_id"]:
        index.append(i)

n_train = int(0.8 * len(index))
is_train = X["profile_id"] <= index[n_train]
is_test = X["profile_id"] > index[n_train]

# conversion to numpy array
X_train = X[is_train].drop(["profile_id"], axis=1).to_numpy()
X_test = X[is_test].drop(["profile_id"], axis=1).to_numpy()
y_train = y[is_train].to_numpy()
y_test = y[is_test].to_numpy()

# %% [markdown]
# We train a toy model. We did not seek to obtain the most accurate model as this notebook is only intended for a proof of concept

# %%
model = Sequential()
model.add(Dense(10, input_dim=X_train.shape[-1]))
model.add(Activation("relu"))
model.add(Dense(20))
model.add(Activation("relu"))
model.add(Dense(1))

# %%
model.compile("adam", "mse")
model.fit(X_train, y_train, batch_size=32, shuffle=True, validation_split=0.2, epochs=3)

# %% [markdown]
# ## Local Robustness to sensoir noise

# %% [markdown]
# In this section, we detail how to derive upper
# and lower bounds on the output of a neural network given some noise on the input.
# Hence we are able to bound formally the worst case prediction given noise.
# In that order, we will use the [decomon](https://gheprivate.intra.corp/CRT-DataScience/decomon/tree/master/decomon) library. Decomon combines several optimization trick, including linear relaxation
# to get state-of-the-art outer approximation.
#
# To use **decomon** for **local robustness to sensor noise** we first need the following imports:
# + *from decomon.models import clone*: to convert our current Keras model into another neural network nn_model. nn_model will output the same prediction that our model and adds extra information that will be used to derive our formal bounds. For a sake of clarity, how to get such bounds is hidden to the user, but an interested reader may refer to
#
#     > _Automatic Perturbation Analysis for Scalable Certified Robustness and Beyond._ NeurIPS 2020. Kaidi Xu*, Zhouxing Shi*, Huan Zhang*, Yihan Wang, Kai-Wei Chang, Minlie Huang, Bhavya Kailkhura, Xue Lin, Cho-Jui Hsieh (* Equal contribution).
#
# + *from decomon import get_lower_noise, get_upper_noise, get_range_noise*: a generic method to get respectively a lower bound, an upper bound, or both on the prediction of a neural network in a $L_p$ (p $\in \{1, 2, \infty\} $) ball with radius epsilon around a sample. If the type of Lp norm is not provided, we assume that we consider a worst case noise independently on every input variable ($L_{\infty}$).

# %%
from decomon import get_lower_noise, get_range_noise, get_upper_noise
from decomon.models import clone
from decomon.perturbation_domain import BallDomain

# %% [markdown]
# ### Noise $L_{\infty}$

# %% [markdown]
# we will first consider a worst case noise independently on every input variable ($L_{\infty}$).
# We pick a random subset of the test dataset and compute an envelop of the network prediction with a noise epsilon

# %%
# you can play with the magnitude of the noise
epsilon = 1e-2

# size of the subset of the test set
n_rand = 1000
# sampling from the test set
index_rand = np.random.permutation(len(X_test))[:n_rand]
X_rand = X_test[index_rand]
y_pred = model.predict(X_rand, verbose=0)[:, 0]

# %% [markdown]
# the **get_upper_noise** and **get_lower_noise methods** return upper and lower bounds over a batchs of samples

# %%
# compute formal bounds
start_time = time.process_time()  # optional
upper_test = get_upper_noise(model, X_rand, eps=epsilon, p=np.inf)[:, 0]
lower_test = get_lower_noise(model, X_rand, eps=epsilon, p=np.inf)[:, 0]
end_time = time.process_time()  # optional

print("Average time to get an upper and a lower bound:{} s".format((end_time - start_time) / n_rand))

# %% [markdown]
# You can compute both bounds within a single call to the method **get_range_noise**

# %%
upper_test_bis, lower_test_bis = get_range_noise(model, X_rand, eps=epsilon, p=np.inf)

# %% [markdown]
# We can assess that the output results remain unchanged

# %%
assert_almost_equal(upper_test, upper_test_bis[:, 0], decimal=4, err_msg="error")
assert_almost_equal(lower_test, lower_test_bis[:, 0], decimal=4, err_msg="error")

# %% [markdown]
# If you plan to compute both upper and lower bounds or call those methods several time in your script, the most efficient way is to call the method on the decomon version itself. To do so, you first need to convert your model:

# %%
start_time = time.process_time()  # optional
perturbation_domain = BallDomain(p=np.inf, eps=epsilon)
nn_model = clone(model, method="crown-forward-hybrid", perturbation_domain=perturbation_domain)
upper_test_ = get_upper_noise(nn_model, X_rand, eps=epsilon, p=np.inf)[:, 0]
lower_test_ = get_lower_noise(nn_model, X_rand, eps=epsilon, p=np.inf)[:, 0]
end_time = time.process_time()  # optional
print("Average time to get an upper and a lower bound:{} s".format((end_time - start_time) / n_rand))

# %% [markdown]
# We can assess that the output results remain unchanged

# %%
assert_almost_equal(upper_test, upper_test_, decimal=4, err_msg="error")
assert_almost_equal(lower_test, lower_test_, decimal=4, err_msg="error")

# %% [markdown]
# Visualization

# %%
plt.plot(np.sort(y_pred), upper_test_[np.argsort(y_pred)], c="k")
plt.plot(np.sort(y_pred), lower_test_[np.argsort(y_pred)], c="b")
plt.plot([y_pred.min(), y_pred.max()], [y_pred.min(), y_pred.max()], "--")
plt.legend(["upper>max NN(x+eps)", "lower<min NN(x+eps)"])
plt.xlabel("predicted temperature (t)")
plt.ylabel("formal bounds given a bounded noise: eps<={}".format(epsilon))
plt.title("Formal robustness in a box")

# %% [markdown]
# ### Noise $L_{2}$
#
# Usually, sensor noise is approximated by Gaussian noise. One way to represent it with formal methods is to use an euclidian ball. We provide an illustration of how to express Gaussian noise in a 2D domain as a pink ball that covers the distribution with high probability.
#
# <img src="./data/ball_fm.png" alt="ball_fm" width="400"/>

# %%
# compute formal bounds
start_time = time.process_time()  # optional
upper_test, lower_test = get_range_noise(model, X_rand, eps=epsilon, p=2)
upper_test = upper_test[:, 0]
lower_test = lower_test[:, 0]
end_time = time.process_time()  # optional

print("Average time to get an upper and a lower bound:{} s".format((end_time - start_time) / n_rand))

# %% [markdown]
# Visualization

# %%
plt.plot(np.sort(y_pred), upper_test[np.argsort(y_pred)], c="k")
plt.plot(np.sort(y_pred), lower_test[np.argsort(y_pred)], c="b")
plt.plot([y_pred.min(), y_pred.max()], [y_pred.min(), y_pred.max()], "--")
plt.legend(["upper>max NN(x+eps)", "lower<min NN(x+eps)"])
plt.xlabel("predicted temperature (t)")
plt.ylabel("formal bounds given a bounded noise: eps<={}".format(epsilon))
plt.title("Formal robustness in an euclidean ball")
