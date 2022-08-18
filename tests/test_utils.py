from __future__ import annotations

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import pytest

from pykronecker.utils import ten

def test_utils():

    with pytest.raises(ValueError):
        ten(np.random.normal(size=(2, 2)))

    with pytest.raises(ValueError):
        ten(np.random.normal(size=2))

    with pytest.raises(ValueError):
        ten(np.random.normal(size=2), shape=(2, 1), like=np.array([1, 2, 4]))