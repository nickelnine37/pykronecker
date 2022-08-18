from __future__ import annotations

import sys
import os

import pytest

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import numpy as np
from utils import assert_universal, generate_test_data, assert_pow, assert_pow_fails, assert_self_hadamard, assert_self_hadamard_fails, assert_inv, assert_inv_fails
from pykronecker.operators import KroneckerProduct, KroneckerDiag


def test_operators():

    print('RUNNING')

    np.set_printoptions(precision=3, linewidth=500, threshold=500, suppress=True, edgeitems=5)

    X, Y, P, kp_literal, ks_literal, kd_literal, kp_optimised, ks_optimised, kd_optimised = generate_test_data()

    assert_universal(X, P, kp_literal, kp_optimised)
    assert_universal(X, P, ks_literal, ks_optimised)
    assert_universal(X, P, kd_literal, kd_optimised)

    assert_inv(kp_literal, kp_optimised)
    assert_inv(kd_literal, kd_optimised)
    assert_inv_fails(ks_optimised)

    assert_pow(kp_literal, kp_optimised)
    assert_pow(kd_literal, kd_optimised)
    assert_pow_fails(ks_optimised)

    assert_self_hadamard(kp_literal, kp_optimised)
    assert_self_hadamard(kd_literal, kd_optimised)
    assert_self_hadamard_fails(ks_optimised)

    # two KroneckerProducts multiplied should give another KroneckerProduct
    assert isinstance(kp_optimised @ kp_optimised, KroneckerProduct)

    # two KroneckerDiags multiplied should give another KroneckerDiag
    assert isinstance(kd_optimised @ kd_optimised, KroneckerDiag)

    with pytest.raises(NotImplementedError):
        kd_optimised ** -1