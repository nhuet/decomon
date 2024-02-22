import pytest
from keras.layers import Dense, Input
from keras.models import Model
from pytest_cases import fixture, parametrize

from decomon.core import ConvertMethod, Propagation, Slope
from decomon.models.convert import clone


def test_clone_nok_several_inputs():
    a = Input((1,))
    b = Input((2,))
    model = Model([a, b], a)

    with pytest.raises(ValueError, match="only 1 input"):
        clone(model)


@parametrize("toy_model_name", ["tutorial"])
def test_clone(
    toy_model_name,
    toy_model_fn,
    method,
    perturbation_domain,
    input_shape,
    simple_model_keras_input_fn,
    simple_model_decomon_input_fn,
    helpers,
):
    slope = Slope.Z_SLOPE
    decimal = 4

    # keras model to convert
    keras_model = toy_model_fn(input_shape=input_shape)

    # symbolic inputs
    keras_symbolic_input = keras_model.inputs[0]

    # actual inputs
    keras_input = simple_model_keras_input_fn(keras_symbolic_input)
    decomon_input = simple_model_decomon_input_fn(keras_input)

    # conversion
    decomon_model = clone(model=keras_model, slope=slope, perturbation_domain=perturbation_domain, method=method)

    # call on actual outputs
    keras_output = keras_model(keras_input)
    decomon_output = decomon_model(decomon_input)

    ibp = decomon_model.ibp
    affine = decomon_model.affine

    if method in (ConvertMethod.FORWARD_IBP, ConvertMethod.FORWARD_HYBRID):
        assert ibp
    else:
        assert not ibp

    if method == ConvertMethod.FORWARD_IBP:
        assert not affine
    else:
        assert affine

    # check ibp and affine bounds well ordered w.r.t. keras inputs/outputs
    helpers.assert_decomon_output_compare_with_keras_input_output_model(
        decomon_output=decomon_output,
        keras_input=keras_input,
        keras_output=keras_output,
        decimal=decimal,
        ibp=ibp,
        affine=affine,
    )
