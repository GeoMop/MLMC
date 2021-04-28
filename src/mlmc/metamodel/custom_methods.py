import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
# import tensorflow.experimental.numpy as tnp
# tnp.experimental_enable_numpy_behavior()
from mlmc.moments import Monomial, Legendre_tf


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
        # quantile = 0.01
        # n_moments = 10
        # domain = np.percentile(y_true, [100 * quantile, 100 * (1 - quantile)])
        # moments_fn_true = Legendre_tf(n_moments, domain)
        #
        # domain = np.percentile(y_predict, [100 * quantile, 100 * (1 - quantile)])
        # moments_fn_predict = Legendre_tf(n_moments, domain)
        moment = 9
        # print("moments_fn.eval_all(y_true).shape ", moments_fn.eval_all(y_true).shape)
        # print("moments_fn.eval(moment, y_true)  ", moments_fn.eval(moment, y_true))
        #print("(moments_fn.eval_all(y_true)", moments_fn.eval_all(y_true)[0])
        #print("moments_fn.eval_all(y_predict) , ", moments_fn_predict.eval_all(y_predict)[0])
        #print("moments_fn.eval_all(y_true)[0] - moments_fn.eval_all(y_predict)[0]", moments_fn.eval_all(y_true)[0] - moments_fn.eval_all(y_predict)[0])

        #print("moments_fn.eval(moment, y_true) - moments_fn.eval(moment, y_predict) ", moments_fn.eval(moment, y_true) - moments_fn.eval(moment, y_predict))
        # moments = [1, 5, 10, 14]
        # sum = 0
        # for m in moments:
        #     sum += K.abs(moments_fn.eval(m, y_true) - moments_fn.eval(m, y_predict)) / 1e10
        #     print("sum ", sum)
        #
        # return K.mean(sum)

        #return K.mean((moments_fn.eval(moment, y_true) - moments_fn.eval(moment, y_predict))**2/1e10)
        #return K.mean(K.sum(K.abs((moments_fn.eval_all(y_true) - moments_fn.eval_all(y_predict))/1e10), axis=1))

        return K.mean(K.sum((moments_fn.eval_all(y_true) - moments_fn.eval_all(y_predict))**2, axis=1))
    return calc_err


def MSE_moments_2(moments_fn=None):
    if moments_fn is None:
        raise ValueError

    def calc_err(y_true, y_predict):
        # quantile = 0.01
        # n_moments = 10
        # domain = np.percentile(y_true, [100 * quantile, 100 * (1 - quantile)])
        # moments_fn_true = Legendre_tf(n_moments, domain)
        #
        # domain = np.percentile(y_predict, [100 * quantile, 100 * (1 - quantile)])
        # moments_fn_predict = Legendre_tf(n_moments, domain)
        moment = 9
        # print("moments_fn.eval_all(y_true).shape ", moments_fn.eval_all(y_true).shape)
        # print("moments_fn.eval(moment, y_true)  ", moments_fn.eval(moment, y_true))
        #print("(moments_fn.eval_all(y_true)", moments_fn.eval_all(y_true)[0])
        #print("moments_fn.eval_all(y_predict) , ", moments_fn_predict.eval_all(y_predict)[0])
        #print("moments_fn.eval_all(y_true)[0] - moments_fn.eval_all(y_predict)[0]", moments_fn.eval_all(y_true)[0] - moments_fn.eval_all(y_predict)[0])

        #print("moments_fn.eval(moment, y_true) - moments_fn.eval(moment, y_predict) ", moments_fn.eval(moment, y_true) - moments_fn.eval(moment, y_predict))

        #return K.mean((moments_fn.eval(moment, y_true) - moments_fn.eval(moment, y_predict))**2)

        moments = [1, 5, 10, 14]
        sum = 0
        for m in moments:
            sum += (moments_fn.eval(m, y_true) - moments_fn.eval(m, y_predict)) ** 2

        return K.mean(sum)

        return K.mean(K.sum((moments_fn.eval_all(y_true) - moments_fn.eval_all(y_predict)) ** 2, axis=1))
        return K.mean(K.sum((moments_fn.eval_all(y_true)[:, 0] - moments_fn.eval_all(y_predict)[:, 0])**2, axis=1))
    return calc_err