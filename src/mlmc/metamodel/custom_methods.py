import tensorflow as tf
from tensorflow.keras import backend as K


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
