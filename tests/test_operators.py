from __future__ import annotations

import sys
import os

import pytest

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import numpy as np
from utils import assert_universal, generate_test_data, assert_pow, assert_pow_fails, assert_self_hadamard, \
    assert_self_hadamard_fails, assert_inv, assert_inv_fails, assert_hadamard, generate_complex_test_data, assert_diag, generate_mixed_test_data1, generate_mixed_test_data2
from pykronecker import KroneckerProduct, KroneckerDiag, KroneckerSum, KroneckerIdentity


def test_operators():

    print('RUNNING')

    np.set_printoptions(precision=3, linewidth=500, threshold=500, suppress=True, edgeitems=5)

    for generator in [generate_test_data, generate_complex_test_data, generate_mixed_test_data1, generate_mixed_test_data2]:

        X, Y, P, kp_literal, ks_literal, kd_literal, kp_optimised, ks_optimised, kd_optimised, ki_literal, ki_optimised = generator()

        # these should all work for kp, kd, ki
        for literal, optimised in zip([kp_literal, kd_literal, ki_literal], [kp_optimised, kd_optimised, ki_optimised]):
            assert_universal(X, P, literal, optimised)
            assert_pow(literal, optimised)
            assert_diag(literal, optimised)
            assert_inv(literal, optimised)

        assert_universal(X, P, ks_literal, ks_optimised)
        assert_diag(ks_literal, ks_optimised)
        assert_inv_fails(ks_optimised)
        assert_pow_fails(ks_optimised)

        # two KroneckerProducts multiplied should give another KroneckerProduct
        assert isinstance(kp_optimised @ kp_optimised, KroneckerProduct)

        # two KroneckerDiags multiplied should give another KroneckerDiag
        assert isinstance(kd_optimised @ kd_optimised, KroneckerDiag)

        for op, op_type in zip([kd_optimised, kp_optimised, ks_optimised, ki_optimised], [KroneckerDiag, KroneckerProduct, KroneckerSum, KroneckerIdentity]):
            assert isinstance(op @ ki_optimised, op_type)
            assert isinstance(ki_optimised @ op, op_type)

        for l1, o1, in zip([kp_literal, ks_literal, kd_literal, ki_literal], [kp_optimised, ks_optimised, kd_optimised, ki_optimised]):
            for l2, o2 in zip([kp_literal, ks_literal, kd_literal, ki_literal], [kp_optimised, ks_optimised, kd_optimised, ki_optimised]):
                assert_hadamard(2 * l1, 2 * o1, 0.5 * l2, 0.5 * o2)
                assert_hadamard(2 * l1, 2 * o1, 0.5 * (l2 + l1), 0.5 * (o2 + o1))

        with pytest.raises(TypeError):
            a = kp_optimised * (kd_optimised @ ks_optimised)

        with pytest.raises(TypeError):
            a = ks_optimised * (kd_optimised @ kp_optimised)

        with pytest.raises(NotImplementedError):
            a = kd_optimised ** -1

        with pytest.raises(NotImplementedError):
            a = ki_optimised ** -1

        with pytest.raises(ValueError):
            a = KroneckerIdentity()
