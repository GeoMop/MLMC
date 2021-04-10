import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow.experimental.numpy as tnp
tnp.experimental_enable_numpy_behavior()
from mlmc.moments import Monomial, Legendre


def abs_activation(x):
    return K.abs(x)


def var_loss_function(y_true, y_predict):
    if tf.is_tensor(y_true):
        y_true = float(y_true)

    # else:
    #     print("diff shape ", (y_true - K.squeeze(y_predict, axis=1)).shape)

    return K.var(y_true - K.squeeze(y_predict, axis=1))
    #return K.sum(K.abs(y_true - K.squeeze(y_predict, axis=1)))


def total_loss_function(y_true, y_predict):
    if tf.is_tensor(y_true):
        y_true = float(y_true)

    # else:
    #     print("diff shape ", (y_true - K.squeeze(y_predict, axis=1)).shape)

    #return K.var(K.abs(y_true - K.squeeze(y_predict, axis=1)))

    return K.mean((y_true - K.squeeze(y_predict, axis=1))**2) + K.var(K.abs(y_true - K.squeeze(y_predict, axis=1)))


def MSE_moments(moments_fn=None):
    if moments_fn is None:
        raise ValueError

    def calc_err(y_true, y_predict):
        return K.mean(K.sum((moments_fn.eval_all(y_true) - moments_fn.eval_all(y_predict))**2, axis=1))
    return calc_err