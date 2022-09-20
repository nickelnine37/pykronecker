from __future__ import annotations
import numpy as np
from numpy import ndarray, diag
from typing import List
from functools import reduce
from typing import Union


numeric = Union[int, float, complex, np.number]


def multiply_tensor_product(As: List[ndarray], X: ndarray) -> ndarray:
    """
    Optimised routine to compute the result of (A1 ⊗ A2 ⊗ ... ⊗ An) @ X
    X can be a tensor of shape (N1, N2, ..., Nn), a vector of shape (N1 x N2 x ... x Nn, )
    or a matrix of shape (N1 x N2 x ... x Nn, [any]), The returned array will have the same shape as X.

    e.g:

    X = randn(2, 3, 4)
    x = randn(2 * 3 * 4)
    P = randn(2 * 3 * 4, 5)
    As = [randn(2, 2), randn(3, 3), randn(4, 4)]

    Y = multiply_tensor_product(As, X)
    y = multiply_tensor_product(As, x)
    Q = multiply_tensor_product(As, P)
    """

    shape = tuple(A.shape[0] for A in As)
    N = int(np.prod(shape))

    # regular tensor
    if X.shape == shape:

        out = X.squeeze()

        for i, A in enumerate(As):
            out = np.tensordot(A, out, axes=[[1], [i]])

        return out.transpose()

    # regular vector
    elif X.squeeze().shape == (N,):

        out = X.squeeze().reshape(shape, order='C')

        for i, A in enumerate(As):
            out = np.tensordot(A, out, axes=[[1], [i]])

        return out.transpose().ravel(order='C').reshape(X.shape)

    # matrix of vector columns
    elif X.ndim == 2 and X.shape[0] == N:

        out = np.zeros(X.shape, dtype=np.result_type(*As, X))

        for j in range(X.shape[1]):

            col = X[:, j].reshape(shape, order='C')

            for i, A in enumerate(As):
                col = np.tensordot(A, col, axes=[[1], [i]])

            out[:, j] = col.transpose().ravel(order='C')

        return out

    else:
        raise ValueError(f'X should have shape {shape} or {(N,)} to match the dimensions of As, but it has shape {X.shape}')


def multiply_tensor_sum(As: List[ndarray], X: ndarray) -> ndarray:
    """
    Optimised routine to compute the result of (A1 ⊕ A2 ⊕ ... ⊕ AN) @ X
    X can be a tensor of shape (N1, N2, ..., Nn), a vector of shape (N1 x N2 x ... x Nn, )
    or a matrix of shape (N1 x N2 x ... x Nn, [any]), The returned array will have the same shape as X.

    e.g:

    X = randn(2, 3, 4)
    x = randn(2 * 3 * 4)
    P = randn(2 * 3 * 4, 5)
    As = [randn(2, 2), randn(3, 3), randn(4, 4)]

    Y = multiply_tensor_sum(As, X)
    y = multiply_tensor_sum(As, x)
    Q = multiply_tensor_sum(As, P)
    """

    shape = tuple(A.shape[0] for A in As)
    N = int(np.prod(shape))

    # regular tensor
    if X.shape == shape:

        out = np.zeros(shape, dtype=np.result_type(*As, X))

        for i, A in enumerate(As):
            trans = list(range(1, len(As)))
            trans.insert(i, 0)
            out += np.tensordot(A, X, axes=[[1], [i]]).transpose(trans)

        return out

    # regular vector
    elif X.squeeze().shape == (N,):

        out = np.zeros(shape, dtype=np.result_type(*As, X))
        X_ = X.squeeze().reshape(shape, order='C')

        for i, A in enumerate(As):
            trans = list(range(1, len(As)))
            trans.insert(i, 0)
            out += np.tensordot(A, X_, axes=[[1], [i]]).transpose(trans)

        return out.ravel(order='C').reshape(X.shape)

    # matrix of vector columns
    elif X.ndim == 2 and X.shape[0] == N:

        out_type = np.result_type(*As, X)
        out = np.zeros(X.shape, dtype=out_type)

        for j in range(X.shape[1]):

            X_ = X[:, j].reshape(shape, order='C')
            col = np.zeros(shape, dtype=out_type)

            for i, A in enumerate(As):
                trans = list(range(1, len(As)))
                trans.insert(i, 0)
                col += np.tensordot(A, X_, axes=[[1], [i]]).transpose(trans)

            out[:, j] = col.ravel(order='C')

        return out

    else:
        raise ValueError(f'X should have shape {shape} or {(N,)} to match the dimensions of As, but it has shape {X.shape}')


def multiply_tensor_diag(A: ndarray, X: ndarray):
    """
    Compute the result of applying a matrix with a diagonal given by the tensor A, to a tensor X.
    X can be a tensor of shape (N1, N2, ..., Nn), a vector of shape (N1 x N2 x ... x Nn, )
    or a matrix of shape (N1 x N2 x ... x Nn, [any]), The returned array will have the same shape as X.

    e.g:

    X = randn(2, 3, 4)
    x = randn(2 * 3 * 4)
    P = randn(2 * 3 * 4, 5)
    A = randn(2, 3, 4)

    Y = multiply_tensor_diag(A, X)
    y = multiply_tensor_diag(A, x)
    Q = multiply_tensor_diag(A, P)
    """

    N = int(np.prod(A.shape))

    # regular tensor
    if X.shape == A.shape:
        return X * A

    # regular vector
    if X.squeeze().shape == (N,):
        return (A.ravel(order='C') * X.squeeze()).reshape(X.shape)

    # matrix of vector columns
    elif X.ndim == 2 and X.shape[0] == N:

        out = np.zeros(X.shape, dtype=np.result_type(A, X))
        A_ = A.ravel(order='C')

        for i in range(X.shape[1]):
            out[:, i] = A_ * X[:, i]

        return out

    else:
        raise ValueError(f'X should have shape {A.shape} or {(N,)} to match the dimensions of A, but it has shape {X.shape}')


def multiply_tensor_identity(shape: tuple, X: ndarray):
    """
    Compute the result of applying the identity matrix to a tensor X.
    X can be a tensor of shape (N1, N2, ..., Nn), a vector of shape (N1 x N2 x ... x Nn, )
    or a matrix of shape (N1 x N2 x ... x Nn, [any]), The returned array will have the same shape as X.

    The main purpose of this simple function is for consistency, and to ensure X has the right shape
    """

    N = shape[0]

    # regular tensor or vector
    if int(np.prod(X.shape)) == N or X.squeeze().shape == (N,):
        return X

    # matrix of vector columns
    elif X.ndim == 2 and X.shape[0] == N:
        return X

    else:
        raise ValueError(f'The product of X.shape should be {(N,)}, but it has shape {X.shape}')


def kronecker_product_literal(As: List[ndarray]) -> ndarray:
    """
    Create an array that is the literal Kronecker product of square matrices As. This should
    never be called for real applications, only used to test the correctness of optimised routines.
    """
    return reduce(np.kron, As)


def kronecker_sum_literal(As: List[ndarray]) -> ndarray:
    """
    Create an array that is the literal Kronecker sum of square matrices As. This should never
    be called for real applications, only used to test the correctness of optimised routines.
    """

    N = int(np.prod([A.shape[0] for A in As]))
    out = np.zeros((N, N), dtype=As[0].dtype)

    for i in range(len(As)):
        Ais = [np.eye(Ai.shape[0]) for Ai in As]
        Ais[i] = As[i]
        out += kronecker_product_literal(Ais)

    return out


def kronecker_diag_literal(X: ndarray) -> ndarray:
    """
    Create an array
    """
    return diag(vec(X))


def vec(X: ndarray) -> ndarray:
    """
    Convert a tensor X of any shape into a vector
    """

    if X.squeeze().ndim == 1:
        return X

    return X.squeeze().ravel(order='C')


def ten(x: ndarray, shape: tuple = None, like: ndarray = None) -> ndarray:
    """
    Convert a vector x into a tensor of a given shape
    """

    if x.shape == shape or (isinstance(like, ndarray) and x.shape == like.shape):
        return x

    if x.squeeze().ndim != 1:
        raise ValueError('x should be 1-dimensional')

    if shape is None and like is None:
        raise ValueError('Pass either shape or like')

    if shape is not None and like is not None:
        raise ValueError('Pass only one of shape or like')

    if shape is not None:
        return x.squeeze().reshape(shape, order='C')

    elif like is not None:
        return x.squeeze().reshape(like.shape, order='C')
