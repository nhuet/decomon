import pytest
import tensorflow.keras.backend as K

from decomon.layers.core import ForwardMode
from decomon.metrics.utils import categorical_cross_entropy


def test_categorical_cross_entropy(odd, mode, floatx, decimal, helpers):
    inputs_0 = helpers.get_tensor_decomposition_multid_box(odd, dc_decomp=False)
    inputs_ = helpers.get_standard_values_multid_box(odd, dc_decomp=False)

    x_0, y_0, z_0, u_c_0, W_u_0, b_u_0, l_c_0, W_l_0, b_l_0 = inputs_0
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    mode = ForwardMode(mode)
    if mode == ForwardMode.HYBRID:
        output = categorical_cross_entropy(inputs_0[2:], dc_decomp=False, mode=mode)
    elif mode == ForwardMode.AFFINE:
        output = categorical_cross_entropy([z_0, W_u_0, b_u_0, W_l_0, b_l_0], dc_decomp=False, mode=mode)
    elif mode == ForwardMode.IBP:
        output = categorical_cross_entropy([u_c_0, l_c_0], dc_decomp=False, mode=mode)
    else:
        raise ValueError("Unknown mode.")

    f_ref = K.function(inputs_0, -y_0 + K.log(K.sum(K.exp(y_0), -1))[:, None])
    f_entropy = K.function(inputs_0, output)

    y_ = f_ref(inputs_)
    if mode == ForwardMode.HYBRID:
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_entropy(inputs_)
        helpers.assert_output_properties_box(
            x, y_, None, None, z_[:, 0], z_[:, 1], u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, decimal=decimal
        )
    elif mode == ForwardMode.AFFINE:
        z_, w_u_, b_u_, w_l_, b_l_ = f_entropy(inputs_)
        helpers.assert_output_properties_box(
            x, y_, None, None, z_[:, 0], z_[:, 1], None, w_u_, b_u_, None, w_l_, b_l_, decimal=decimal
        )
    elif mode == ForwardMode.IBP:
        u_c_, l_c_ = f_entropy(inputs_)
        helpers.assert_output_properties_box(
            x, y_, None, None, inputs_[2][:, 0], inputs_[2][:, 1], u_c_, None, None, l_c_, None, None, decimal=decimal
        )
    else:
        raise ValueError("Unknown mode.")
