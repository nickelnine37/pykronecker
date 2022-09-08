from __future__ import annotations

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import pytest

from utils import assert_universal, generate_test_data

np.set_printoptions(precision=3, linewidth=500, threshold=500, suppress=True, edgeitems=5)


def test_sum():

    X, Y, P, kp_literal, ks_literal, kd_literal, kp_optimised, ks_optimised, kd_optimised, ki_literal, ki_optimised = generate_test_data()

    literal1 = kp_literal + ks_literal
    literal2 = kd_literal - ks_literal / 2
    literal3 = 2.5 * kp_literal + ks_literal / 3 + kd_literal

    optimised1 = kp_optimised + ks_optimised
    optimised2 = kd_optimised - ks_optimised / 2
    optimised3 = 2.5 * kp_optimised + ks_optimised / 3 + kd_optimised

    assert_universal(X, P, literal1, optimised1)
    assert_universal(X, P, literal2, optimised2)
    assert_universal(X, P, literal3, optimised3)

    # Operator sums cannot be inverted or powered
    for optimised in [optimised1, optimised2, optimised3]:

        with pytest.raises(NotImplementedError):
            optimised.inv()

        with pytest.raises(NotImplementedError):
            optimised ** 2



def test_product():

    X, Y, P, kp_literal, ks_literal, kd_literal, kp_optimised, ks_optimised, kd_optimised, ki_literal, ki_optimised = generate_test_data()

    literal1 = kp_literal @ ks_literal
    literal2 = kd_literal @ ks_literal / 2
    literal3 = 2.5 * kp_literal @ ks_literal / 3 + kd_literal

    optimised1 = kp_optimised @ ks_optimised
    optimised2 = kd_optimised @ ks_optimised / 2
    optimised3 = 2.5 * kp_optimised @ ks_optimised / 3 + kd_optimised

    assert_universal(X, P, literal1, optimised1)
    assert_universal(X, P, literal2, optimised2)
    assert_universal(X, P, literal3, optimised3)

    # operator products can be inverted if all suboperators are invertible
    assert np.allclose(np.linalg.inv(2 * kp_literal @ kd_literal), (2 * kp_optimised @ kd_optimised).inv().to_array())

    # operator products cannot be powered
    for optimised in [optimised1, optimised2, optimised3]:

        with pytest.raises(NotImplementedError):
            optimised ** 2


