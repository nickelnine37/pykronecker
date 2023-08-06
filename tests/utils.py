from __future__ import annotations

import pytest
from numpy import ndarray
import numpy as np

from pykronecker.base import KroneckerOperator
from pykronecker import KroneckerProduct, KroneckerDiag, KroneckerSum
from pykronecker.operators import KroneckerIdentity
from pykronecker.utils import kronecker_product_literal, kronecker_sum_literal, kronecker_diag_literal, vec, ten

import jax
import jax.numpy as jnp


def generate_test_data(seed: int=0,
                       matrix_kind: str='numpy',
                       tensor_kind: str='numpy',
                       matrix_type: str='real',
                       tensor_type: str='real'):
    """
    Generate random data for testing purposes.
    """

    np.random.seed(seed)

    N1 = 5
    N2 = 4
    N3 = 3
    K = 3

    if matrix_kind == 'numpy':

        A1 = np.random.randn(N1, N1)
        A2 = np.random.randn(N2, N2)
        A3 = np.random.randn(N3, N3)
        D = np.random.randn(N1, N2, N3)

    elif matrix_kind == 'jax':

        key = jax.random.PRNGKey(seed)

        A1 = jax.random.normal(key, (N1, N1), dtype=jnp.float32)
        A2 = jax.random.normal(key, (N2, N2), dtype=jnp.float32)
        A3 = jax.random.normal(key, (N3, N3), dtype=jnp.float32)
        D = jax.random.normal(key, (N1, N2, N3), dtype=jnp.float32)

    else:
        raise ValueError(f"matrix_type should be 'numpy' or 'jax' but it is {matrix_kind}")

    if matrix_type == 'real':
        pass

    elif matrix_type == 'complex':

        A1 = A1 + 1j * A1.T
        A2 = A2 + 1j * A2.T
        A3 = A3 + 1j * A3.T
        D = D + 1j * D

    else:
        raise ValueError("matrix_type should be 'real' or 'complex' ")

    if tensor_kind == 'numpy':

        X = np.random.randn(N1, N2, N3)
        Y = np.random.randn(N1, N2, N3)
        Q = np.random.randn(N1 * N2 * N3, K)

    elif tensor_kind == 'jax':

        key = jax.random.PRNGKey(seed)

        X = jax.random.normal(key, (N1, N2, N3), dtype=jnp.float32)
        Y = jax.random.normal(key, (N1, N2, N3), dtype=jnp.float32)
        Q = jax.random.normal(key, (N1 * N2 * N3, K), dtype=jnp.float32)

    else:
        raise ValueError(f"tensor_kind should be 'numpy' or 'jax' but it is {tensor_kind}")

    if tensor_type == 'real':
        pass

    elif tensor_type == 'complex':

        X = X + 1j * X
        Y = Y + 1j * Y
        Q = Q + 1j * Q

    else:
        raise ValueError(f"tensor_kind should be one of ['numpy', 'jax'] but it is {tensor_kind}")

    # create actual array structures
    kp_literal = kronecker_product_literal([A1, A2, A3])
    ks_literal = kronecker_sum_literal([A1, A2, A3])
    kd_literal = kronecker_diag_literal(D)
    ki_literal = np.eye(N1 * N2 * N3)

    kp_optimised = KroneckerProduct([A1, A2, A3])
    ks_optimised = KroneckerSum([A1, A2, A3])
    kd_optimised = KroneckerDiag(D)
    ki_optimised = KroneckerIdentity(tensor_shape=(N1, N2, N3))

    return X, Y, Q, kp_literal, ks_literal, kd_literal, kp_optimised, ks_optimised, kd_optimised, ki_literal, ki_optimised


def assert_close(a: ndarray, b: ndarray, shape: tuple):
    assert np.allclose(a, b, rtol=1e-2, atol=1e-4), f'failed: MSE = {((a - b) ** 2).sum() / np.prod(shape):.4e}. literal = {a.ravel()}, optimised = {b.ravel()}'


def assert_conversions(literal: ndarray, optimised: KroneckerOperator):
    """
    Test basic conversions
    """

    a, b = literal, optimised.to_array()
    assert_close(a, b, literal.shape)

    a, b = literal.T, optimised.T.to_array()
    assert_close(a, b, literal.shape)

    a, b = literal.conj(), optimised.conj().to_array()
    assert_close(a, b, literal.shape)

    a, b = literal.conj().T, optimised.H.to_array()
    assert_close(a, b, literal.shape)

    a, b = literal, (+optimised).to_array()
    assert_close(a, b, literal.shape)

    a, b = -literal, (-optimised).to_array()
    assert_close(a, b, literal.shape)


def assert_indexing(literal: ndarray, optimised: KroneckerOperator):
    """
    Assert all supported indexing functions work
    """

    a, b = optimised[2], literal[2]
    assert_close(a, b, literal.shape)

    a, b = optimised[(2,)], literal[(2,)]
    assert_close(a, b, literal.shape)

    a, b = optimised[2:5], literal[2:5]
    assert_close(a, b, literal.shape)

    a, b = optimised[2:8:3], literal[2:8:3]
    assert_close(a, b, literal.shape)

    a, b = optimised[:, 2], literal[:, 2]
    assert_close(a, b, literal.shape)

    a, b = optimised[:, 2:5], literal[:, 2:5]
    assert_close(a, b, literal.shape)

    a, b = optimised[:, 2:8:3], literal[:, 2:8:3]
    assert_close(a, b, literal.shape)

    a, b = optimised[2, :], literal[2, :]
    assert_close(a, b, literal.shape)

    a, b = optimised[2:5, :], literal[2:5, :]
    assert_close(a, b, literal.shape)

    a, b = optimised[2:8:3, :], literal[2:8:3, :]
    assert_close(a, b, literal.shape)

    a, b = optimised[2:5, 2:8:3], literal[2:5, 2:8:3]
    assert_close(a, b, literal.shape)

    a, b = optimised[2, 5], literal[2, 5]
    assert_close(a, b, literal.shape)

    a, b = optimised[:], literal[:]
    assert_close(a, b, literal.shape)

    a, b = optimised[:, :], literal[:, :]
    assert_close(a, b, literal.shape)

    with pytest.raises(IndexError):
        optimised[2, 3, 4]

    with pytest.raises(IndexError):
        optimised['hey']

    with pytest.raises(IndexError):
        optimised['a', 'b']


def assert_basic_fails(optimised: KroneckerOperator):

    with pytest.raises(TypeError):
        optimised + 1

    with pytest.raises(TypeError):
        1 + optimised

    with pytest.raises(TypeError):
        optimised @ 5

    with pytest.raises(TypeError):
        optimised * '5'

    with pytest.raises(ValueError):
        optimised.sum(2)


def assert_vec_matrix_multiply(X: ndarray, literal: ndarray, optimised: KroneckerOperator):
    """
    Assert forwards and backwards vec matrix multiplication
    """

    a, b = literal @ vec(X), optimised @ vec(X)
    assert_close(a, b, X.shape)

    a, b = vec(X) @ literal, vec(X) @ optimised
    assert_close(a, b, X.shape)

    a, b = vec(X) @ literal @ vec(X), vec(X) @ optimised @ vec(X)
    assert_close(a, b, X.shape)


def assert_multivec_matrix_multiply(P: ndarray, literal: ndarray, optimised: KroneckerOperator):
    """
    Assert forwards and backwards vec matrix multiplication
    """
    
    a, b = literal @ P, optimised @ P
    assert_close(a, b, P.shape)

    
    a, b = P.T @ literal, P.T @ optimised
    assert_close(a, b, P.shape)

    
    a, b = P.T @ literal @ P, P.T @ optimised @ P
    assert_close(a, b, P.shape)



def assert_ten_matrix_multiply(X: ndarray, literal: ndarray, optimised: KroneckerOperator):
    """
    Assert forwards and backwards ten matrix multiplication
    """

    a, b = ten(literal @ vec(X), like=X), optimised @ X
    assert_close(a, b, X.shape)


    a, b = ten(vec(X) @ literal, shape=X.shape), X @ optimised
    assert_close(a, b, X.shape)



def assert_np_matmul(X: ndarray, literal: ndarray, optimised: KroneckerOperator):
    """
    Assert forwards and backwards matrix multiplication with np.matmul
    """
    a, b = np.matmul(literal, vec(X)), np.matmul(optimised, vec(X))
    assert_close(a, b, X.shape)

    
    a, b = np.matmul(vec(X), literal), np.matmul(vec(X), optimised)
    assert_close(a, b, X.shape)

    
    a, b = np.matmul(vec(X), np.matmul(literal, vec(X))), np.matmul(vec(X), np.matmul(optimised, vec(X)))
    assert_close(a, b, X.shape)


def assert_sum(literal: ndarray, optimised: KroneckerOperator):
    """
    Test summing operations
    """

    a, b = literal.sum(0), optimised.sum(0),
    assert_close(a, b, literal.shape)

    a, b = literal.sum(1), optimised.sum(1)
    assert_close(a, b, literal.shape)

    a, b = literal.sum(), optimised.sum()
    assert_close(a, b, literal.shape)


def assert_diag(literal: ndarray, optimised: KroneckerOperator):
    """
    Test diag operation
    """
    a, b = np.diag(literal), optimised.diag()
    assert_close(a, b, a.shape)


def assert_scalar_multiply(literal: ndarray, optimised: KroneckerOperator):
    """
    Test scalar multiply operations with ints, floats, np.floats and comlpex numbers
    """

    for factor in [-1, 0, np.random.normal(), np.float64(np.random.normal()), np.random.normal() + 1j * np.random.normal()]:
        assert np.allclose(factor * literal, (factor * optimised).to_array(), rtol=1e-2, atol=1e-4)
        assert np.allclose(literal * factor, (optimised * factor).to_array(), rtol=1e-2, atol=1e-4)
        assert np.allclose(np.multiply(factor, literal), np.multiply(factor, optimised).to_array(), rtol=1e-2, atol=1e-4)
        assert np.allclose(np.multiply(literal, factor), np.multiply(optimised, factor).to_array(), rtol=1e-2, atol=1e-4)


def assert_str(optimised: KroneckerOperator):
    assert optimised.__str__()
    assert optimised.__repr__()


def assert_copy(literal: ndarray, optimised: KroneckerOperator):
    assert np.allclose(literal, optimised.copy().to_array(), rtol=1e-2, atol=1e-4)
    assert np.allclose(literal, optimised.deepcopy().to_array(), rtol=1e-2, atol=1e-4)


def assert_exceptions(optimised: KroneckerOperator):

    with pytest.raises(NotImplementedError):
        np.divide(optimised, 5)

    with pytest.raises(TypeError):
        optimised @ 5

    with pytest.raises(TypeError):
        5 @ optimised


def assert_pow(literal: ndarray, optimised: KroneckerOperator):

    for power in [1.0, 3, 2.0]:
        assert np.allclose(literal ** power, (optimised ** power).to_array())


def assert_pow_fails(optimised: KroneckerOperator):

    for power in [1.0, 3, 2.0]:

        with pytest.raises(NotImplementedError):
            optimised ** power


def assert_self_hadamard(literal: ndarray, optimised: KroneckerOperator):

    a, b = literal * literal, (optimised * optimised).to_array()
    assert_close(a, b, literal.shape)


def assert_hadamard(literal1: ndarray, optimised1: KroneckerOperator, literal2: ndarray, optimised2: KroneckerOperator):

    a, b = literal1 * literal2, (optimised1 * optimised2).to_array()
    assert_close(a, b, literal1.shape)


def assert_self_hadamard_fails(optimised: KroneckerOperator):

    with pytest.raises(TypeError):
        optimised * optimised


def assert_inv(literal: ndarray, optimised: KroneckerOperator):

    a, b = np.linalg.inv(literal), optimised.inv().to_array()
    assert_close(a, b, literal.shape)


def assert_inv_fails(optimised: KroneckerOperator):

    with pytest.raises(NotImplementedError):
        optimised.inv()


def assert_shape_fails(optimised: KroneckerOperator):

    x = np.random.randn(int(np.prod(optimised.shape) + 1))

    with pytest.raises(ValueError):
        optimised @ x


def assert_universal(X: ndarray, P: ndarray, literal: ndarray, optimised: KroneckerOperator):
    """
    Assert that the ndarray `literal` and the KroneckerOperator `optimised` behave in the exact same
    way when applied to the tensor X, and matrix of vectors P.
    """

    # basic functionality
    assert_conversions(literal, optimised)
    assert_indexing(literal, optimised)
    assert_str(optimised)
    assert_copy(literal, optimised)
    assert_basic_fails(optimised)
    assert isinstance(optimised.tensor_shape, tuple)

    # matrix multiplications
    assert_vec_matrix_multiply(X, literal, optimised)
    assert_ten_matrix_multiply(X, literal, optimised)
    assert_np_matmul(X, literal, optimised)
    assert_multivec_matrix_multiply(P, literal, optimised)
    assert_shape_fails(optimised)

    # math
    assert_sum(literal, optimised)
    assert_scalar_multiply(literal, optimised)
    assert_exceptions(optimised)



