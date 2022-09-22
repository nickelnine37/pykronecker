from __future__ import annotations
import numpy as np
from numpy import ndarray, diag
from typing import List, Union
from functools import reduce

numeric = Union[int, float, complex, np.number]


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

    if x.shape == shape or (like is not None and x.shape == like.shape):
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
