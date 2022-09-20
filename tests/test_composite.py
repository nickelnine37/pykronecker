from __future__ import annotations

import numpy as np
import sys
import os

from pykronecker import KroneckerProduct, KroneckerDiag

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import pytest

from utils import assert_universal, generate_test_data, generate_complex_test_data, assert_diag, assert_inv, generate_mixed_test_data1, generate_mixed_test_data2

np.set_printoptions(precision=3, linewidth=500, threshold=500, suppress=True, edgeitems=5)


def test_sum():

    for generator in [generate_test_data, generate_complex_test_data, generate_mixed_test_data1, generate_mixed_test_data2]:

        X, Y, P, kp_literal, ks_literal, kd_literal, kp_optimised, ks_optimised, kd_optimised, ki_literal, ki_optimised = generator()

        literal1 = kp_literal + ks_literal
        literal2 = kd_literal - ks_literal / 2
        literal3 = 2.5 * kp_literal + ks_literal / 3 + kd_literal

        optimised1 = kp_optimised + ks_optimised
        optimised2 = kd_optimised - ks_optimised / 2
        optimised3 = 2.5 * kp_optimised + ks_optimised / 3 + kd_optimised

        for literal, optimised in zip([literal1, literal2, literal3], [optimised1, optimised2, optimised3]):

            assert_universal(X, P, literal, optimised)
            assert_diag(literal, optimised)

            with pytest.raises(NotImplementedError):
                optimised.inv()

            with pytest.raises(NotImplementedError):
                optimised ** 2



def test_product():

    for generator in [generate_test_data, generate_complex_test_data, generate_mixed_test_data1, generate_mixed_test_data2]:

        X, Y, P, kp_literal, ks_literal, kd_literal, kp_optimised, ks_optimised, kd_optimised, ki_literal, ki_optimised = generator()

        literal1 = kp_literal @ ks_literal
        literal2 = kd_literal @ ks_literal / 2
        literal3 = 2.5 * kp_literal @ ks_literal / 3 + kd_literal

        optimised1 = kp_optimised @ ks_optimised
        optimised2 = kd_optimised @ ks_optimised / 2
        optimised3 = 2.5 * kp_optimised @ ks_optimised / 3 + kd_optimised

        for literal, optimised in zip([literal1, literal2, literal3], [optimised1, optimised2, optimised3]):

            assert_universal(X, P, literal, optimised)
            assert_diag(literal, optimised)

            with pytest.raises(NotImplementedError):
                optimised ** 2

        # these should not return operator products
        assert isinstance(kp_optimised @ kp_optimised, KroneckerProduct)
        assert isinstance(kd_optimised @ kd_optimised, KroneckerDiag)

        # operator products can be inverted if all sub-operators are invertible
        assert_inv(2 * kd_literal @ kp_literal, 2 * kd_optimised @ kp_optimised)





