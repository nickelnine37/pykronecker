from __future__ import annotations
import numpy as np
from numpy import ndarray, diag
from typing import List


def multiply_tensor_product(As: List[ndarray], X: ndarray) -> ndarray:
    """
    Optimised routine to compute the result of Ten((A1 ⊗ A2 ⊗ ... ⊗ AN) vec(X))

    e.g:

    X = randn(2, 3, 4)
    A1 = randn(4, 4); A2 = randn(3, 3); A3 = randn(2, 2)
    X_ = tensor_product([A1, A2, A3], X)
    """

    assert X.ndim == len(As), f'Input was expected to be {len(As)}-dimensional, but it was {X.ndim}-dimensional'
    assert all(A.shape == (s, s) for A, s in zip(As, reversed(X.shape))), f'Input was expected to have shape {tuple(A.shape[0] for A in As[::-1])} but it has shape {X.shape}'

    ans = X

    for i, A in enumerate(reversed(As)):
        ans = np.tensordot(A, ans, axes=[[1], [i]])

    return ans.transpose()


def multiply_tensor_sum(As: List[ndarray], X: ndarray) -> ndarray:
    """
    Optimised routine to compute the result of Ten((A1 ⊕ A2 ⊕ ... ⊕ AN) vec(X))

    e.g:

    X = randn(2, 3, 4)
    A1 = randn(4, 4); A2 = randn(3, 3); A3 = randn(2, 2)
    X_ = tensor_product_of_sum([A1, A2, A3], X)
    """

    assert X.ndim == len(As), f'Input was expected to be {len(As)}-dimensional, but it was {X.ndim}-dimensional'
    assert all(A.shape == (s, s) for A, s in zip(As, reversed(X.shape))), f'Input was expected to have shape {tuple(A.shape[0] for A in As[::-1])} but it has shape {X.shape}'

    ans = np.zeros_like(X)

    for i, A in enumerate(reversed(As)):
        trans = list(range(1, len(As)))
        trans.insert(i, 0)
        ans += np.tensordot(A, X, axes=[[1], [i]]).transpose(trans)

    return ans


def kronecker_product_literal(As: List[ndarray]) -> ndarray:
    """
    Create an array that is the literal Kronecker product of square matrices As. This should
    never be called for real applications, only used to test the correctness of more optimised
    routines.
    """
    if len(As) == 2:
        return np.kron(As[0], As[1])
    else:
        return np.kron(As[0], kronecker_product_literal(As[1:]))


def kronecker_sum_literal(As: List[ndarray]) -> ndarray:
    """
    Create an array that is the literal Kronecker sum of square matrices As. This should never
    be called for real applications, only used to test the correctness of optimised routines.
    """
    tot = 0.0
    for i in range(len(As)):
        Ais = [np.eye(len(Ai)) for Ai in As]
        Ais[i] = As[i]
        tot += kronecker_product_literal(Ais)

    return tot


def kronecker_diag_literal(X: ndarray) -> ndarray:
    """
    Create an array
    """
    return diag(vec(X))


def vec(X: ndarray) -> ndarray:
    """
    Convert a tensor X of any shape into a vector
    """
    if X.ndim == 1:
        return X
    return X.reshape(-1, order='F')


def ten(x: ndarray, shape: tuple = None, like: ndarray = None) -> ndarray:
    """
    Convert a vector x into a tensor of a given shape
    """

    if x.shape == shape or (isinstance(like, ndarray) and x.shape == like.shape):
        return x

    if x.ndim != 1:
        raise ValueError('x should be 1-dimensional')

    if shape is None and like is None:
        raise ValueError('Pass either shape or like')

    if shape is not None and like is not None:
        raise ValueError('Pass only one of shape or like')

    if shape is not None:
        return x.reshape(shape, order='F')

    elif like is not None:
        return x.reshape(like.shape, order='F')
