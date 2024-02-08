"""Run benchmark on decomon dense layer

Usage:
    run_dense.py [--naive --backward --1d]

Options:
    --naive:    if present, use the naive implementation
    --backward: if present, run backward propagation, else forward propagation
    --1d:       if present, flatten inputs

"""


from time import perf_counter

import keras.ops as K
import numpy as np
from conftest import BoxDomain, Helpers, Propagation
from docopt import docopt
from keras.layers import Dense, Input

from decomon.keras_utils import batch_multid_dot
from decomon.layers.core.dense import DecomonDense, DecomonNaiveDense

perturbation_domain = BoxDomain()
ibp, affine = True, True


def main(decomon_layer_class, propagation, input_shape, batchsize, units, testcase_name):
    output_shape = input_shape[:-1] + (units,)

    layer = Dense(units=units)
    layer(Input(input_shape))

    keras_input = K.ones((batchsize,) + input_shape)
    layer(keras_input)

    decomon_layer = decomon_layer_class(
        layer=layer, ibp=ibp, affine=affine, propagation=propagation, perturbation_domain=perturbation_domain
    )

    decomon_input_shape = Helpers.get_decomon_input_shapes(
        model_input_shape=input_shape,
        model_output_shape=output_shape,
        layer_input_shape=input_shape,
        layer_output_shape=output_shape,
        ibp=ibp,
        affine=affine,
        propagation=propagation,
        perturbation_domain=perturbation_domain,
    )
    affine_bounds_to_propagate_shape, constant_oracle_bounds_shape, x_shape = decomon_input_shape
    x = K.ones((batchsize,) + x_shape)
    affine_bounds_to_propagate = [K.ones((batchsize,) + shape) for shape in affine_bounds_to_propagate_shape]
    constant_oracle_bounds = [K.ones((batchsize,) + shape) for shape in constant_oracle_bounds_shape]
    decomon_input = [affine_bounds_to_propagate, constant_oracle_bounds, x]

    start = perf_counter()
    decomon_layer(*decomon_input)
    duration = perf_counter() - start

    print(f"{testcase_name}: {duration}")


if __name__ == "__main__":
    #  arguments
    arguments = docopt(__doc__)
    naive = arguments["--naive"]
    backward = arguments["--backward"]
    flatten = arguments["--1d"]

    if naive:
        decomon_layer_class = DecomonNaiveDense
    else:
        decomon_layer_class = DecomonDense

    if backward:
        propagation = Propagation.BACKWARD
    else:
        propagation = Propagation.FORWARD

    input_shape = (2, 7, 8, 10)
    batchsize = 50
    units = 18

    if flatten:
        input_shape = (np.prod(input_shape[:-1]), input_shape[-1])

    testcase_name = f"{decomon_layer_class.__name__}-{propagation.value}{'-1d' if flatten else ''}"
    main(
        decomon_layer_class,
        propagation,
        input_shape=input_shape,
        batchsize=batchsize,
        units=units,
        testcase_name=testcase_name,
    )
