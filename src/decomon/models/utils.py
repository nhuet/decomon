from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
import tensorflow as tf
import tensorflow.python.keras.backend as K
from keras.engine.node import Node
from tensorflow.keras.layers import (
    Concatenate,
    Flatten,
    Input,
    Lambda,
    Layer,
    Maximum,
    Minimum,
)
from tensorflow.keras.models import Model

from decomon.layers.core import ForwardMode
from decomon.utils import (
    ConvexDomainType,
    get_lower,
    get_lower_layer,
    get_upper,
    get_upper_layer,
)


class ConvertMethod(Enum):
    CROWN = "crown"
    CROWN_FORWARD_IBP = "crown-forward-ibp"
    CROWN_FORWARD_AFFINE = "crown-forward-affine"
    CROWN_FORWARD_HYBRID = "crown-forward-hybrid"
    FORWARD_IBP = "forward-ibp"
    FORWARD_AFFINE = "forward-affine"
    FORWARD_HYBRID = "forward-hybrid"


def check_input_tensors_sequential(
    model: Model,
    input_tensors: Optional[List[tf.Tensor]],
    input_dim: int,
    input_dim_init: int,
    IBP: bool,
    forward: bool,
    dc_decomp: bool,
    convex_domain: Optional[Dict[str, Any]],
) -> List[tf.Tensor]:

    if convex_domain is None:
        convex_domain = {}
    if input_tensors is None:  # no predefined input tensors

        input_shape = list(K.int_shape(model.input)[1:])

        if len(convex_domain) == 0 and not isinstance(input_dim, tuple):
            input_dim_ = (2, input_dim)
            z_tensor = Input(input_dim_, dtype=model.layers[0].dtype)
        elif convex_domain["name"] == ConvexDomainType.BOX and not isinstance(input_dim, tuple):
            input_dim_ = (2, input_dim)
            z_tensor = Input(input_dim_, dtype=model.layers[0].dtype)
        else:

            if isinstance(input_dim, tuple):
                z_tensor = Input(input_dim, dtype=model.layers[0].dtype)
            else:
                z_tensor = Input((input_dim,), dtype=model.layers[0].dtype)

        if dc_decomp:
            h_tensor = Input(tuple(input_shape), dtype=model.layers[0].dtype)
            g_tensor = Input(tuple(input_shape), dtype=model.layers[0].dtype)
        if forward:
            b_u_tensor = Input(tuple(input_shape), dtype=model.layers[0].dtype)
            b_l_tensor = Input(tuple(input_shape), dtype=model.layers[0].dtype)
            if input_dim_init > 0:
                w_u_tensor = Input(tuple([input_dim] + input_shape), dtype=model.layers[0].dtype)
                w_l_tensor = Input(tuple([input_dim] + input_shape), dtype=model.layers[0].dtype)
            else:
                w_u_tensor = Input(tuple(input_shape), dtype=model.layers[0].dtype)
                w_l_tensor = Input(tuple(input_shape), dtype=model.layers[0].dtype)

        if IBP:
            u_c_tensor = Input(tuple(input_shape), dtype=model.layers[0].dtype)
            l_c_tensor = Input(tuple(input_shape), dtype=model.layers[0].dtype)
            if forward:  # hybrid mode
                input_tensors = [
                    z_tensor,
                    u_c_tensor,
                    w_u_tensor,
                    b_u_tensor,
                    l_c_tensor,
                    w_l_tensor,
                    b_l_tensor,
                ]
            else:
                # only IBP
                input_tensors = [
                    u_c_tensor,
                    l_c_tensor,
                ]
        elif forward:
            # forward mode
            input_tensors = [
                z_tensor,
                w_u_tensor,
                b_u_tensor,
                w_l_tensor,
                b_l_tensor,
            ]
        else:
            raise NotImplementedError("not IBP and not forward not implemented")

        if dc_decomp:
            input_tensors += [h_tensor, g_tensor]

    else:
        # assert that input_tensors is a List of 6 InputLayer objects
        # If input tensors are provided, the original model's InputLayer is
        # overwritten with a different InputLayer.
        assert isinstance(input_tensors, list), "expected input_tensors to be a List or None, but got {}".format(
            type(input_tensors)
        )

        if dc_decomp:
            if IBP and forward:
                assert len(input_tensors) == 9, "wrong number of inputs, expexted 10 but got {}".format(
                    len(input_tensors)
                )
            if IBP and not forward:
                assert len(input_tensors) == 4, "wrong number of inputs, expexted 6 but got {}".format(
                    len(input_tensors)
                )
            if not IBP and forward:
                assert len(input_tensors) == 5, "wrong number of inputs, expexted 8 but got {}".format(
                    len(input_tensors)
                )
        else:
            if IBP and forward:
                assert len(input_tensors) == 7, "wrong number of inputs, expexted 10 but got {}".format(
                    len(input_tensors)
                )
            if IBP and not forward:
                assert len(input_tensors) == 2, "wrong number of inputs, expexted 10 but got {}".format(
                    len(input_tensors)
                )
            if not IBP and forward:
                assert len(input_tensors) == 5, "wrong number of inputs, expexted 10 but got {}".format(
                    len(input_tensors)
                )

    return input_tensors


def get_input_tensor_x(
    model: Model,
    input_tensors: Optional[List[tf.Tensor]],
    input_dim: int,
    input_dim_init: int,
    convex_domain: Dict[str, Any],
) -> Input:
    if len(convex_domain) == 0 and not isinstance(input_dim, tuple):
        input_dim_ = (2, input_dim)
        z_tensor = Input(input_dim_, dtype=model.layers[0].dtype)
    elif convex_domain["name"] == ConvexDomainType.BOX and not isinstance(input_dim, tuple):
        input_dim_ = (2, input_dim)
        z_tensor = Input(input_dim_, dtype=model.layers[0].dtype)
    else:

        if isinstance(input_dim, tuple):
            z_tensor = Input(input_dim, dtype=model.layers[0].dtype)
        else:
            z_tensor = Input((input_dim,), dtype=model.layers[0].dtype)
    return z_tensor


def check_input_tensors_functionnal(
    model: Model,
    input_tensors: Optional[List[tf.Tensor]],
    input_dim: int,
    input_dim_init: int,
    IBP: bool,
    forward: bool,
    dc_decomp: bool,
    convex_domain: Optional[Dict[str, Any]],
) -> List[tf.Tensor]:

    raise NotImplementedError()


def get_mode(IBP: bool = True, forward: bool = True) -> ForwardMode:

    if IBP:
        if forward:
            return ForwardMode.HYBRID
        else:
            return ForwardMode.IBP
    else:
        return ForwardMode.AFFINE


def get_depth_dict(model: Model) -> Dict[int, List[Node]]:

    depth_keys = list(model._nodes_by_depth.keys())
    depth_keys.sort(reverse=True)

    nodes_list = []

    dico_depth: Dict[int, int] = {}
    dico_nodes: Dict[int, List[Node]] = {}

    def fill_dico(node: Node, dico_depth: Optional[Dict[int, int]] = None) -> Dict[int, int]:
        if dico_depth is None:
            dico_depth = {}

        parents = node.parent_nodes
        if len(parents):
            for parent in parents:
                if id(parent) not in dico_depth:
                    dico_depth = fill_dico(parent, dico_depth)

                depth = np.min([dico_depth[id(parent)] - 1 for parent in parents])
                if id(node) in dico_depth:
                    dico_depth[id(node)] = max(depth, dico_depth[id(node)])
                else:
                    dico_depth[id(node)] = depth
        else:
            dico_depth[id(node)] = max(depth_keys)

        return dico_depth

    for depth in depth_keys:
        # check for nodes that do not have parents and set depth to maximum depth value
        nodes = model._nodes_by_depth[depth]
        nodes_list += nodes
        for node in nodes:
            dico_depth = fill_dico(node, dico_depth)

    for node in nodes_list:
        depth = dico_depth[id(node)]
        if depth in dico_nodes:
            dico_nodes[depth].append(node)
        else:
            dico_nodes[depth] = [node]

    return dico_nodes


def get_inner_layers(model: Model) -> int:

    count = 0
    for layer in model.layers:
        if isinstance(layer, Model):
            count += get_inner_layers(layer)
        else:
            count += 1
    return count


def convert_2_mode(
    mode_from: Union[str, ForwardMode],
    mode_to: Union[str, ForwardMode],
    convex_domain: Optional[Dict[str, Any]],
    dtype: Union[str, tf.DType] = K.floatx(),
) -> Layer:
    mode_from = ForwardMode(mode_from)
    mode_to = ForwardMode(mode_to)

    def get_2_mode_priv(inputs_: List[tf.Tensor]) -> List[tf.Tensor]:

        if mode_from == mode_to:
            return inputs_

        if mode_from in [ForwardMode.AFFINE, ForwardMode.HYBRID]:
            x_0 = inputs_[0]
        elif mode_from == ForwardMode.IBP:
            u_c, l_c = inputs_
            if mode_to in [ForwardMode.AFFINE, ForwardMode.HYBRID]:
                x_0 = K.concatenate([K.expand_dims(l_c, 1), K.expand_dims(u_c, 1)], 1)
                z_value = K.cast(0.0, u_c.dtype)
                o_value = K.cast(1.0, u_c.dtype)
                w = tf.linalg.diag(z_value * l_c)
                # b = z_value * l_c + o_value
                w_u = w
                b_u = u_c
                w_l = w
                b_l = l_c
            else:
                raise ValueError(f"Unknown mode {mode_to}")
        else:
            raise ValueError(f"Unknown mode {mode_from}")

        if mode_from == ForwardMode.AFFINE:
            _, w_u, b_u, w_l, b_l = inputs_
            if mode_to in [ForwardMode.IBP, ForwardMode.HYBRID]:
                u_c = get_upper(x_0, w_u, b_u, convex_domain=convex_domain)
                l_c = get_lower(x_0, w_l, b_l, convex_domain=convex_domain)
        elif mode_from == ForwardMode.IBP:
            u_c, l_c = inputs_
        elif mode_from == ForwardMode.HYBRID:
            _, u_c, w_u, b_u, l_c, w_l, b_l = inputs_
        else:
            raise ValueError(f"Unknown mode {mode_from}")

        if mode_to == ForwardMode.IBP:
            return [u_c, l_c]
        elif mode_to == ForwardMode.AFFINE:
            return [x_0, w_u, b_u, w_l, b_l]
        elif mode_to == ForwardMode.HYBRID:
            return [x_0, u_c, w_u, b_u, l_c, w_l, b_l]
        raise ValueError(f"Unknown mode {mode_to}")

    return Lambda(get_2_mode_priv, dtype=dtype)
