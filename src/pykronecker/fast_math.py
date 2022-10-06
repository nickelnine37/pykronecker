from __future__ import annotations
import numpy as np
from numpy import ndarray
from typing import List


try:
    import jax.numpy as jnp
    from jax import jit
    import jax
    use_jax = True
    print(f'Using Jax backend with device {jax.devices()[0]}')

except ImportError:
    use_jax = False
    print('Using NumPy backend')


mod = jnp if use_jax else np


def kronecker_product_tensor(As: List[ndarray], X: ndarray):
    """
    Apply the Kronecker product of square matrices As to tensor X
    """

    out = X

    for A in As:
        out = (A @ out.reshape(A.shape[0], -1)).T

    return out.reshape(X.shape)


def kronecker_product_vector(As: List[ndarray], X: ndarray, shape: tuple):
    """
    Apply the Kronecker product of square matrices As to vector X
    """
    return kronecker_product_tensor(As, X.reshape(shape)).ravel()


def kronecker_product_vector_columns(As: List[ndarray], X: ndarray, shape: tuple):
    """
    Apply the Kronecker product of square matrices As to matrix of vector columns X
    """
    return mod.vstack([kronecker_product_vector(As, X[:, j], shape) for j in range(X.shape[1])]).T


def get_trans(i: int, n: int):
    """
    Helper function for transposing in kronecker_sum_tensor
    """
    trans = list(range(1, n))
    trans.insert(i, 0)
    return trans


def kronecker_sum_tensor(As: List[ndarray], X: ndarray):
    """
    Apply the Kronecker sum of square matrices As to tensor X
    """
    # return sum((A @ X.transpose(np.roll(range(3), -i)).reshape(A.shape[0], -1)).reshape(np.array(X.shape)[np.roll(range(3), -i)]).transpose(np.roll(range(3), i)) for i, A in enumerate(As))
    # ^ avoids tensordot but not as fast
    return sum(mod.tensordot(A, X, axes=[[1], [i]]).transpose(get_trans(i, len(As))) for i, A in enumerate(As))


def kronecker_sum_vector(As: List[ndarray], X: ndarray, shape: tuple):
    """
    Apply the Kronecker sum of square matrices As to vector X
    """

    return kronecker_sum_tensor(As, X.reshape(shape)).ravel()


def kronecker_sum_vector_columns(As: List[ndarray], X: ndarray, shape: tuple):
    """
    Apply the Kronecker sum of square matrices As to matrix of vector columns X
    """

    return mod.vstack([kronecker_sum_vector(As, X[:, j], shape) for j in range(X.shape[1])]).T


def kronecker_diag_tensor(A: ndarray, X: ndarray):
    """
    Apply the Kronecker diag of tensor A to tensor X
    """
    return X * A


def kronecker_diag_vector(A: ndarray, X: ndarray):
    """
    Apply the Kronecker diag of tensor A to vector X
    """
    return kronecker_diag_tensor(A.ravel(), X.squeeze()).reshape(X.shape)


def kronecker_diag_vector_columns(A: ndarray, X: ndarray):
    """
    Apply the Kronecker diag of tensor A to vector X
    """
    return mod.vstack([kronecker_diag_vector(A, X[:, j]) for j in range(X.shape[1])]).T


# apply jit wrapper to all tensor functions
if use_jax:

    kronecker_product_tensor = jit(kronecker_product_tensor)
    kronecker_sum_tensor = jit(kronecker_sum_tensor)
    kronecker_diag_tensor = jit(kronecker_diag_tensor)


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
        return kronecker_product_tensor(As, X)

    # regular vector
    elif X.squeeze().shape == (N,):
        return kronecker_product_vector(As, X, shape)

    # matrix of vector columns
    elif X.ndim == 2 and X.shape[0] == N:
        return kronecker_product_vector_columns(As, X, shape)

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
        return kronecker_sum_tensor(As, X)

    # regular vector
    elif X.squeeze().shape == (N,):
        return kronecker_sum_vector(As, X, shape)

    # matrix of vector columns
    elif X.ndim == 2 and X.shape[0] == N:
        return kronecker_sum_vector_columns(As, X, shape)

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
        return kronecker_diag_tensor(A, X)

    # regular vector
    if X.squeeze().shape == (N,):
        return kronecker_diag_vector(A, X)

    # matrix of vector columns
    elif X.ndim == 2 and X.shape[0] == N:
        return kronecker_diag_vector_columns(A, X)

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

    if (int(np.prod(X.shape)) == N) or (X.squeeze().shape == (N,)) or (X.ndim == 2 and X.shape[0] == N):
        return X

    else:
        raise ValueError(f'The product of X.shape should be {(N,)}, but it has shape {X.shape}')
