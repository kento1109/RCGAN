'''
MMD functions implemented in tensorflow.
(from https://github.com/dougalsutherland/opt-mmd/blob/master/gan/mmd.py)
'''
from __future__ import division

import tensorflow as tf

from tf_ops import dot, sq_sum

from scipy.spatial.distance import pdist
from numpy import median, vstack, einsum
import pdb
import numpy as np
from pprint import pprint

_eps=1e-8

### additions from sugimoto, for convenience
sigma_opt_iter = 2000
sigma_opt_thresh = 0.001

################################################################################
### Quadratic-time MMD with Gaussian RBF kernel

def _mix_rbf_kernel(X, Y, sigmas, wts=None):
    """
    """
    if wts is None:
        wts = [1.0] * sigmas.get_shape()[0]

    # debug!
    if len(X.shape) == 2:
        # matrix
        XX = tf.matmul(X, X, transpose_b=True)
        XY = tf.matmul(X, Y, transpose_b=True)
        YY = tf.matmul(Y, Y, transpose_b=True)
    elif len(X.shape) == 3:
        # tensor -- this is computing the Frobenius norm
        XX = tf.tensordot(X, X, axes=[[1, 2], [1, 2]])
        XY = tf.tensordot(X, Y, axes=[[1, 2], [1, 2]])
        YY = tf.tensordot(Y, Y, axes=[[1, 2], [1, 2]])
    else:
        raise ValueError(X)

    X_sqnorms = tf.diag_part(XX)
    Y_sqnorms = tf.diag_part(YY)

    r = lambda x: tf.expand_dims(x, 0)
    c = lambda x: tf.expand_dims(x, 1)

    K_XX, K_XY, K_YY = 0, 0, 0
    for sigma, wt in zip(tf.unstack(sigmas, axis=0), wts):
        gamma = 1 / (2 * sigma**2)
        K_XX += wt * tf.exp(-gamma * (-2 * XX + c(X_sqnorms) + r(X_sqnorms)))
        K_XY += wt * tf.exp(-gamma * (-2 * XY + c(X_sqnorms) + r(Y_sqnorms)))
        K_YY += wt * tf.exp(-gamma * (-2 * YY + c(Y_sqnorms) + r(Y_sqnorms)))

    return K_XX, K_XY, K_YY, tf.reduce_sum(wts)


def rbf_mmd2(X, Y, sigma=1, biased=True):
    return mix_rbf_mmd2(X, Y, sigmas=[sigma], biased=biased)


def mix_rbf_mmd2(X, Y, sigmas=(1,), wts=None, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigmas, wts)
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)


def rbf_mmd2_and_ratio(X, Y, sigma=1, biased=True):
    return mix_rbf_mmd2_and_ratio(X, Y, sigmas=[sigma], biased=biased)


def mix_rbf_mmd2_and_ratio(X, Y, sigmas=(1,), wts=None, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigmas, wts)
    return _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)


################################################################################
### Helper functions to compute variances based on kernel matrices


def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = tf.cast(K_XX.get_shape()[0], tf.float32)
    n = tf.cast(K_YY.get_shape()[0], tf.float32)

    if biased:
        mmd2 = (tf.reduce_sum(K_XX) / (m * m)
              + tf.reduce_sum(K_YY) / (n * n)
              - 2 * tf.reduce_sum(K_XY) / (m * n))
    else:
        if const_diagonal is not False:
            trace_X = m * const_diagonal
            trace_Y = n * const_diagonal
        else:
            trace_X = tf.trace(K_XX)
            trace_Y = tf.trace(K_YY)

        mmd2 = ((tf.reduce_sum(K_XX) - trace_X) / (m * (m - 1))
              + (tf.reduce_sum(K_YY) - trace_Y) / (n * (n - 1))
              - 2 * tf.reduce_sum(K_XY) / (m * n))

    return mmd2


def _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=False, biased=False,
                    min_var_est=_eps):
    mmd2, var_est = _mmd2_and_variance(
        K_XX, K_XY, K_YY, const_diagonal=const_diagonal, biased=biased)
    ratio = mmd2 / tf.sqrt(tf.maximum(var_est, min_var_est))
    return mmd2, ratio


def _mmd2_and_variance(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = tf.cast(K_XX.get_shape()[0], tf.float32)  # Assumes X, Y are same shape

    ### Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if const_diagonal is not False:
        const_diagonal = tf.cast(const_diagonal, tf.float32)
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
        sum_diag2_X = sum_diag2_Y = m * const_diagonal**2
    else:
        diag_X = tf.diag_part(K_XX)
        diag_Y = tf.diag_part(K_YY)

        sum_diag_X = tf.reduce_sum(diag_X)
        sum_diag_Y = tf.reduce_sum(diag_Y)

        sum_diag2_X = sq_sum(diag_X)
        sum_diag2_Y = sq_sum(diag_Y)

    Kt_XX_sums = tf.reduce_sum(K_XX, 1) - diag_X
    Kt_YY_sums = tf.reduce_sum(K_YY, 1) - diag_Y
    K_XY_sums_0 = tf.reduce_sum(K_XY, 0)
    K_XY_sums_1 = tf.reduce_sum(K_XY, 1)

    Kt_XX_sum = tf.reduce_sum(Kt_XX_sums)
    Kt_YY_sum = tf.reduce_sum(Kt_YY_sums)
    K_XY_sum = tf.reduce_sum(K_XY_sums_0)

    Kt_XX_2_sum = sq_sum(K_XX) - sum_diag2_X
    Kt_YY_2_sum = sq_sum(K_YY) - sum_diag2_Y
    K_XY_2_sum  = sq_sum(K_XY)

    if biased:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
              + (Kt_YY_sum + sum_diag_Y) / (m * m)
              - 2 * K_XY_sum / (m * m))
    else:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * (m-1))
              + (Kt_YY_sum + sum_diag_Y) / (m * (m-1))
              - 2 * K_XY_sum / (m * m))

    var_est = (
          2 / (m**2 * (m-1)**2) * (
              2 * sq_sum(Kt_XX_sums) - Kt_XX_2_sum
            + 2 * sq_sum(Kt_YY_sums) - Kt_YY_2_sum)
        - (4*m-6) / (m**3 * (m-1)**3) * (Kt_XX_sum**2 + Kt_YY_sum**2)
        + 4*(m-2) / (m**3 * (m-1)**2) * (
              sq_sum(K_XY_sums_1) + sq_sum(K_XY_sums_0))
        - 4 * (m-3) / (m**3 * (m-1)**2) * K_XY_2_sum
        - (8*m - 12) / (m**5 * (m-1)) * K_XY_sum**2
        + 8 / (m**3 * (m-1)) * (
              1/m * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
            - dot(Kt_XX_sums, K_XY_sums_1)
            - dot(Kt_YY_sums, K_XY_sums_0))
    )

    return mmd2, var_est


### additions from stephanie, for convenience

def median_pairwise_distance(X, Y=None):
    """
    Heuristic for bandwidth of the RBF. Median pairwise distance of joint data.
    If Y is missing, just calculate it from X:
        this is so that, during training, as Y changes, we can use a fixed
        bandwidth (and save recalculating this each time we evaluated the mmd)
    At the end of training, we do the heuristic "correctly" by including
    both X and Y.

    Note: most of this code is assuming tensorflow, but X and Y are just ndarrays
    """
    if Y is None:
        Y = X       # this is horrendously inefficient, sorry
   
    if len(X.shape) == 2:
        # matrix
        X_sqnorms = einsum('...i,...i', X, X)
        Y_sqnorms = einsum('...i,...i', Y, Y)
        XY = einsum('ia,ja', X, Y)
    elif len(X.shape) == 3:
        # tensor -- this is computing the Frobenius norm
        X_sqnorms = einsum('...ij,...ij', X, X)
        Y_sqnorms = einsum('...ij,...ij', Y, Y)
        XY = einsum('iab,jab', X, Y)
    else:
        raise ValueError(X)

    distances = np.sqrt(X_sqnorms.reshape(-1, 1) - 2*XY + Y_sqnorms.reshape(1, -1))
    return median(distances)

### additions from sugimoto, for convenience

def tf_initialize(x_eval, seq_length, input_dim):
    # get heuristic bandwidth for mmd kernel from evaluation samples
    heuristic_sigma_training = median_pairwise_distance(x_eval)
    # optimise sigma using that (that's t-hat)
    eval_size = x_eval.shape[0]
    eval_eval_size = int(0.1 * eval_size)
    eval_real_PH = tf.placeholder(tf.float32, [eval_eval_size, seq_length, input_dim])
    eval_sample_PH = tf.placeholder(tf.float32, [eval_eval_size, seq_length, input_dim])
    n_sigmas = 2
    sigma = tf.get_variable(name='sigma', shape=n_sigmas, initializer=tf.constant_initializer(
        value=np.power(heuristic_sigma_training, np.linspace(-1, 3, num=n_sigmas))))
    mmd2, that = mix_rbf_mmd2_and_ratio(eval_real_PH, eval_sample_PH, sigma)
    with tf.variable_scope("SIGMA_optimizer"):
        # sigma_solver = tf.train.RMSPropOptimizer(learning_rate=0.05).minimize(-that, var_list=[sigma])
        sigma_solver = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(-that, var_list=[sigma])
        # sigma_solver = tf.train.AdagradOptimizer(learning_rate=0.1).minimize(-that, var_list=[sigma])
    sigma_opt_vars = [var for var in tf.global_variables() if 'SIGMA_optimizer' in var.name]
    dic_tf_sigma = {'sigma_opt_vars': sigma_opt_vars,
                    'sigma_solver': sigma_solver,
                    'eval_real_PH': eval_real_PH,
                    'eval_sample_PH': eval_sample_PH}
    # session start
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    return sigma, dic_tf_sigma, that, sess


def sigma_optimization(eval_real, eval_sample, sigma, dic_tf_sigma, that, sess):
    eval_size = eval_real.shape[0]
    eval_eval_size = int(0.1 * eval_size)
    eval_eval_real = eval_real[:eval_eval_size]
    eval_test_real = eval_real[eval_eval_size:]
    eval_eval_sample = eval_sample[:eval_eval_size]
    eval_test_sample = eval_sample[eval_eval_size:]
    
    # print(eval_test_real.shape)

    sess.run(tf.variables_initializer(dic_tf_sigma['sigma_opt_vars']))
    sigma_iter = 0
    that_change = sigma_opt_thresh * 2
    old_that = 0
    while that_change > sigma_opt_thresh and sigma_iter < sigma_opt_iter:
        new_sigma, that_np, _ = sess.run([sigma, that,
                                          dic_tf_sigma['sigma_solver']],
                                         feed_dict={dic_tf_sigma['eval_real_PH']: eval_eval_real,
                                                    dic_tf_sigma['eval_sample_PH']: eval_eval_sample})
        that_change = np.abs(that_np - old_that)
        old_that = that_np
        sigma_iter += 1
    opt_sigma = sess.run(sigma)
    mmd2, that_np = sess.run(mix_rbf_mmd2_and_ratio(eval_test_real,
                                                    eval_test_sample,
                                                    biased=False,
                                                    sigmas=sigma))
    return mmd2, that_np


