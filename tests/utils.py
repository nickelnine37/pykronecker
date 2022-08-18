import pytest
from numpy import ndarray
import numpy as np

from pykronecker.base import KroneckerOperator
from pykronecker.operators import KroneckerProduct, KroneckerDiag, KroneckerSum
from pykronecker.utils import kronecker_product_literal, kronecker_sum_literal, kronecker_diag_literal, vec, ten


def generate_test_data(seed: int = 0):
    """
    Generate random data for testing purposes
    """

    np.random.seed(seed)

    N1 = 6
    N2 = 5
    N3 = 4
    N4 = 3
    K = 5

    A1 = np.random.randn(N1, N1)
    A2 = np.random.randn(N2, N2)
    A3 = np.random.randn(N3, N3)
    A4 = np.random.randn(N4, N4)
    D = np.random.randn(N4, N3, N2, N1)

    X = np.random.randn(N4, N3, N2, N1)
    Y = np.random.randn(N4, N3, N2, N1)
    Q = np.random.randn(N4 * N3 * N2 * N1, K)

    # create actual array structures
    kp_literal = kronecker_product_literal([A1, A2, A3, A4])
    ks_literal = kronecker_sum_literal([A1, A2, A3, A4])
    kd_literal = kronecker_diag_literal(D)

    kp_optimised = KroneckerProduct([A1, A2, A3, A4])
    ks_optimised = KroneckerSum([A1, A2, A3, A4])
    kd_optimised = KroneckerDiag(D)

    return X, Y, Q, kp_literal, ks_literal, kd_literal, kp_optimised, ks_optimised, kd_optimised


def assert_conversions(literal: ndarray, optimised: KroneckerOperator):
    """
    Test basic conversions
    """
    assert np.allclose(literal, optimised.to_array())
    assert np.allclose(literal.T, optimised.T.to_array())
    assert np.allclose(literal, (+optimised).to_array())
    assert np.allclose(-literal, (-optimised).to_array())


def assert_basic_fails(optimised: KroneckerOperator):

    with pytest.raises(TypeError):
        optimised + 1

    with pytest.raises(TypeError):
        1 + optimised

    with pytest.raises(TypeError):
        optimised @ 5

    with pytest.raises(TypeError):
        optimised * '5'

    with pytest.raises(TypeError):
        optimised.quadratic_form(5)

    with pytest.raises(ValueError):
        optimised.sum(2)

def assert_vec_matrix_multiply(X: ndarray, literal: ndarray, optimised: KroneckerOperator):
    """
    Assert forwards and backwards vec matrix multiplication
    """

    assert np.allclose(literal @ vec(X), optimised @ vec(X))  # test forward matrix multiplication
    assert np.allclose(vec(X) @ literal, vec(X) @ optimised)  # test backward matrix multiplication
    assert np.isclose(vec(X) @ literal @ vec(X), vec(X) @ optimised @ vec(X))  # test quadratic form


def assert_multivec_matrix_multiply(P: ndarray, literal: ndarray, optimised: KroneckerOperator):
    """
    Assert forwards and backwards vec matrix multiplication
    """
    assert np.allclose(literal @ P, optimised @ P)
    assert np.allclose(P.T @ literal, P.T @ optimised)
    assert np.allclose(P.T @ literal @ P, P.T @ optimised @ P)


def assert_ten_matrix_multiply(X: ndarray, literal: ndarray, optimised: KroneckerOperator):
    """
    Assert forwards and backwards ten matrix multiplication
    """
    assert np.allclose(ten(literal @ vec(X), like=X), optimised @ X)  # test forward matrix multiplication
    assert np.allclose(ten(vec(X) @ literal, like=X), X @ optimised)  # test backward matrix multiplication


def assert_np_matmul(X: ndarray, literal: ndarray, optimised: KroneckerOperator):
    """
    Assert forwards and backwards matrix multiplication with np.matmul
    """
    assert np.allclose(np.matmul(literal, vec(X)), np.matmul(optimised, vec(X)))
    assert np.allclose(np.matmul(vec(X), literal), np.matmul(vec(X), optimised))
    assert np.isclose(np.matmul(vec(X), np.matmul(literal, vec(X))), np.matmul(vec(X), np.matmul(optimised, vec(X))))


def assert_sum(literal: ndarray, optimised: KroneckerOperator):
    """
    Test summing operations
    """
    assert np.allclose(literal.sum(0), optimised.sum(0))
    assert np.allclose(literal.sum(1), optimised.sum(1))
    assert np.isclose(literal.sum(), optimised.sum())


def assert_scalar_multiply(literal: ndarray, optimised: KroneckerOperator):
    """
    Test summing operations
    """

    for factor in [-1, 0, np.random.normal(), np.float64(np.random.normal())]:
        assert np.allclose(factor * literal, (factor * optimised).to_array())
        assert np.allclose(literal * factor, (optimised * factor).to_array())
        assert np.allclose(np.multiply(factor, literal), np.multiply(factor, optimised).to_array())
        assert np.allclose(np.multiply(literal, factor), np.multiply(optimised, factor).to_array())


def assert_str(optimised: KroneckerOperator):
    assert optimised.__str__()
    assert optimised.__repr__()


def assert_copy(literal: ndarray, optimised: KroneckerOperator):
    assert np.allclose(literal, optimised.copy().to_array())
    assert np.allclose(literal, optimised.deepcopy().to_array())


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
    assert np.allclose(literal * literal, (optimised * optimised).to_array())

def assert_self_hadamard_fails(optimised: KroneckerOperator):

    with pytest.raises(TypeError):
        optimised * optimised

def assert_inv(literal: ndarray, optimised: KroneckerOperator):
    assert np.allclose(np.linalg.inv(literal), optimised.inv().to_array())


def assert_inv_fails(optimised: KroneckerOperator):

    with pytest.raises(NotImplementedError):
        optimised.inv()



def assert_universal(X: ndarray, P: ndarray, literal: ndarray, optimised: KroneckerOperator):
    """
    Assert that the ndarray `literal` and the KroneckerOperator `optimised` behave in the exact same
    way when applied to the tensor X, and matrix of vectors P.
    """

    # basic functionality
    assert_conversions(literal, optimised)
    assert_str(optimised)
    assert_copy(literal, optimised)
    assert_basic_fails(optimised)

    # matrix multiplications
    assert_vec_matrix_multiply(X, literal, optimised)
    assert_ten_matrix_multiply(X, literal, optimised)
    assert_np_matmul(X, literal, optimised)
    assert_multivec_matrix_multiply(P, literal, optimised)

    # math
    assert_sum(literal, optimised)
    assert_scalar_multiply(literal, optimised)
    assert_exceptions(optimised)



