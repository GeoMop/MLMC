import copy
import warnings

import numpy as np
from scipy import linalg
from scipy import sparse as sp
from scipy.sparse.linalg import ArpackNoConvergence

import tensorflow as tf
from tensorflow.keras import backend as K

from spektral.layers import ops
from spektral.layers.convolutional.conv import Conv
import matplotlib.pyplot as plt


class OwnChebConv(Conv):
    r"""
    A Chebyshev convolutional layer from the paper

    > [Convolutional Neural Networks on Graphs with Fast Localized Spectral
  Filtering](https://arxiv.org/abs/1606.09375)<br>
    > Michaël Defferrard et al.

    **Mode**: single, disjoint, mixed, batch.

    This layer computes:
    $$
        \X' = \sum \limits_{k=0}^{K - 1} \T^{(k)} \W^{(k)}  + \b^{(k)},
    $$
    where \( \T^{(0)}, ..., \T^{(K - 1)} \) are Chebyshev polynomials of \(\tilde \L\)
    defined as
    $$
        \T^{(0)} = \X \\
        \T^{(1)} = \tilde \L \X \\
        \T^{(k \ge 2)} = 2 \cdot \tilde \L \T^{(k - 1)} - \T^{(k - 2)},
    $$
    where
    $$
        \tilde \L =  \frac{2}{\lambda_{max}} \cdot (\I - \D^{-1/2} \A \D^{-1/2}) - \I.
    $$

    **Input**

    - Node features of shape `([batch], n_nodes, n_node_features)`;
    - A list of K Chebyshev polynomials of shape
    `[([batch], n_nodes, n_nodes), ..., ([batch], n_nodes, n_nodes)]`; can be computed with
    `spektral.utils.convolution.chebyshev_filter`.

    **Output**

    - Node features with the same shape of the input, but with the last
    dimension changed to `channels`.

    **Arguments**

    - `channels`: number of output channels;
    - `K`: order of the Chebyshev polynomials;
    - `activation`: activation function;
    - `use_bias`: bool, add a bias vector to the output;
    - `kernel_initializer`: initializer for the weights;
    - `bias_initializer`: initializer for the bias vector;
    - `kernel_regularizer`: regularization applied to the weights;
    - `bias_regularizer`: regularization applied to the bias vector;
    - `activity_regularizer`: regularization applied to the output;
    - `kernel_constraint`: constraint applied to the weights;
    - `bias_constraint`: constraint applied to the bias vector.

    """

    def __init__(
        self,
        channels,
        K=1,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super().__init__(
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )
        self.channels = channels
        self.K = K

        # self.use_bias = use_bias

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[0][-1]

        self.kernel = self.add_weight(
            shape=(self.K, input_dim, self.channels),
            initializer=self.kernel_initializer,
            name="kernel",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.channels,),
                initializer=self.bias_initializer,
                name="bias",
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        x, a = inputs

        T_0 = x
        output = K.dot(T_0, self.kernel[0])

        if self.K > 1:
            T_1 = ops.modal_dot(a, x)
            output += K.dot(T_1, self.kernel[1])

            # print("T_1 ", T_1)
            # print("self.kernel[1] ", self.kernel[1])

        for k in range(2, self.K):
            T_2 = 2 * ops.modal_dot(a, T_1) - T_0
            output += K.dot(T_2, self.kernel[k])
            T_0, T_1 = T_1, T_2

        #print("self use bias ", self.use_bias)
        if self.use_bias:
            #print("use bias")
            output = K.bias_add(output, self.bias)
        output = self.activation(output)

        return output

    @property
    def config(self):
        return {"channels": self.channels, "K": self.K}

    @staticmethod
    def preprocess(a):
        a = normalized_laplacian(a)
        a = rescale_laplacian(a)
        return a


def degree_matrix(A):
    """
    Computes the degree matrix of the given adjacency matrix.
    :param A: rank 2 array or sparse matrix.
    :return: if A is a dense array, a dense array; if A is sparse, a sparse
    matrix in DIA format.
    """
    degrees = np.array(A.sum(1)).flatten()
    if sp.issparse(A):
        D = sp.diags(degrees)
    else:
        D = np.diag(degrees)
    return D


def degree_power(A, k):
    r"""
    Computes \(\D^{k}\) from the given adjacency matrix. Useful for computing
    normalised Laplacian.
    :param A: rank 2 array or sparse matrix.
    :param k: exponent to which elevate the degree matrix.
    :return: if A is a dense array, a dense array; if A is sparse, a sparse
    matrix in DIA format.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        degrees = np.power(np.array(A.sum(1)), k).ravel()
    degrees[np.isinf(degrees)] = 0.0
    if sp.issparse(A):
        D = sp.diags(degrees)
    else:
        D = np.diag(degrees)
    return D


def normalized_adjacency(A, symmetric=True):
    r"""
    Normalizes the given adjacency matrix using the degree matrix as either
    \(\D^{-1}\A\) or \(\D^{-1/2}\A\D^{-1/2}\) (symmetric normalization).
    :param A: rank 2 array or sparse matrix;
    :param symmetric: boolean, compute symmetric normalization;
    :return: the normalized adjacency matrix.
    """
    if symmetric:
        #print("symmetric")
        normalized_D = degree_power(A, -0.5)
        #print("normalized D")
        return normalized_D.dot(A).dot(normalized_D)
    else:
        normalized_D = degree_power(A, -1.0)
        return normalized_D.dot(A)


def laplacian(A):
    r"""
    Computes the Laplacian of the given adjacency matrix as \(\D - \A\).
    :param A: rank 2 array or sparse matrix;
    :return: the Laplacian.
    """
    return degree_matrix(A) - A


def normalized_laplacian(A, symmetric=True):
    r"""
    Computes a  normalized Laplacian of the given adjacency matrix as
    \(\I - \D^{-1}\A\) or \(\I - \D^{-1/2}\A\D^{-1/2}\) (symmetric normalization).
    :param A: rank 2 array or sparse matrix;
    :param symmetric: boolean, compute symmetric normalization;
    :return: the normalized Laplacian.
    """
    if sp.issparse(A):
        I = sp.eye(A.shape[-1], dtype=A.dtype)
    else:
        I = np.eye(A.shape[-1], dtype=A.dtype)
    normalized_adj = normalized_adjacency(A, symmetric=symmetric)
    return I - normalized_adj


def rescale_laplacian(L, lmax=None):
    """
    Rescales the Laplacian eigenvalues in [-1,1], using lmax as largest eigenvalue.
    :param L: rank 2 array or sparse matrix;
    :param lmax: if None, compute largest eigenvalue with scipy.linalg.eisgh.
    If the eigendecomposition fails, lmax is set to 2 automatically.
    If scalar, use this value as largest eigenvalue when rescaling.
    :return:
    """
    if lmax is None:
        try:
            if sp.issparse(L):
                lmax = sp.linalg.eigsh(L, 1, which="LM", return_eigenvectors=False)[0]
            else:
                n = L.shape[-1]
                lmax = linalg.eigh(L, eigvals_only=True, eigvals=[n - 2, n - 1])[-1]
        except ArpackNoConvergence:
            lmax = 2
    if sp.issparse(L):
        I = sp.eye(L.shape[-1], dtype=L.dtype)
    else:
        I = np.eye(L.shape[-1], dtype=L.dtype)
    L_scaled = (2.0 / lmax) * L - I
    return L_scaled


def gcn_filter(A, symmetric=True):
    r"""
    Computes the graph filter described in
    [Kipf & Welling (2017)](https://arxiv.org/abs/1609.02907).
    :param A: array or sparse matrix with rank 2 or 3;
    :param symmetric: boolean, whether to normalize the matrix as
    \(\D^{-\frac{1}{2}}\A\D^{-\frac{1}{2}}\) or as \(\D^{-1}\A\);
    :return: array or sparse matrix with rank 2 or 3, same as A;
    """
    out = copy.deepcopy(A)
    if isinstance(A, list) or (isinstance(A, np.ndarray) and A.ndim == 3):
        for i in range(len(A)):
            out[i] = A[i]
            out[i][np.diag_indices_from(out[i])] += 1
            out[i] = normalized_adjacency(out[i], symmetric=symmetric)
    else:
        if hasattr(out, "tocsr"):
            out = out.tocsr()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out[np.diag_indices_from(out)] += 1
        out = normalized_adjacency(out, symmetric=symmetric)

    if sp.issparse(out):
        out.sort_indices()
    return out


def chebyshev_polynomial(X, k):
    """
    Calculates Chebyshev polynomials of X, up to order k.
    :param X: rank 2 array or sparse matrix;
    :param k: the order up to which compute the polynomials,
    :return: a list of k + 1 arrays or sparse matrices with one element for each
    degree of the polynomial.
    """
    T_k = list()
    if sp.issparse(X):
        T_k.append(sp.eye(X.shape[0], dtype=X.dtype).tocsr())
    else:
        T_k.append(np.eye(X.shape[0], dtype=X.dtype))
    T_k.append(X)

    def chebyshev_recurrence(T_k_minus_one, T_k_minus_two, X):
        if sp.issparse(X):
            X_ = sp.csr_matrix(X, copy=True)
        else:
            X_ = np.copy(X)
        return 2 * X_.dot(T_k_minus_one) - T_k_minus_two

    for _ in range(2, k + 1):
        T_k.append(chebyshev_recurrence(T_k[-1], T_k[-2], X))

    return T_k


def chebyshev_filter(A, k, symmetric=True):
    r"""
    Computes the Chebyshev filter from the given adjacency matrix, as described
    in [Defferrard et at. (2016)](https://arxiv.org/abs/1606.09375).
    :param A: rank 2 array or sparse matrix;
    :param k: integer, the order of the Chebyshev polynomial;
    :param symmetric: boolean, whether to normalize the matrix as
    \(\D^{-\frac{1}{2}}\A\D^{-\frac{1}{2}}\) or as \(\D^{-1}\A\);
    :return: a list of k + 1 arrays or sparse matrices with one element for each
    degree of the polynomial.
    """
    normalized_adj = normalized_adjacency(A, symmetric)
    if sp.issparse(A):
        I = sp.eye(A.shape[0], dtype=A.dtype)
    else:
        I = np.eye(A.shape[0], dtype=A.dtype)
    L = I - normalized_adj  # Compute Laplacian

    # Rescale Laplacian
    L_scaled = rescale_laplacian(L)

    # Compute Chebyshev polynomial approximation
    T_k = chebyshev_polynomial(L_scaled, k)

    # Sort indices
    if sp.issparse(T_k[0]):
        for i in range(len(T_k)):
            T_k[i].sort_indices()

    return T_k


def add_self_loops(a, value=1):
    """
    Sets the inner diagonals of `a` to `value`.
    :param a: a np.array or scipy.sparse matrix, the innermost two dimensions
    must be equal.
    :param value: value to set the diagonals to.
    :return: a np.array or scipy.sparse matrix with the same shape as `a`.
    """
    a = a.copy()
    if len(a.shape) < 2:
        raise ValueError("a must have at least rank 2")
    n = a.shape[-1]
    if n != a.shape[-2]:
        raise ValueError(
            "Innermost two dimensions must be equal. Got {}".format(a.shape)
        )
    if sp.issparse(a):
        a = a.tolil()
        a.setdiag(value)
        return a.tocsr()
    else:
        idx = np.arange(n)
        a[..., idx, idx] = value
        return a
