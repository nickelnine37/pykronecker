import unittest
import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from utils import BaseTestCases

from pykronecker import KroneckerSum
from pykronecker.utils import kronecker_sum_literal


class TestKroneckerSum11(BaseTestCases.BaseTest):

    def get_operators(self):
        As = [self.get_array(shape, self.matrix_type, self.matrix_kind) for shape in self.shapes]
        return KroneckerSum(As), kronecker_sum_literal(As)

    def test_inv(self):
        self.assertRaises(NotImplementedError, lambda: self.A_opt.inv())

    def test_pow(self):
        self.assertRaises(NotImplementedError, lambda: self.A_opt ** 2)


class TestKroneckerSum12(TestKroneckerSum11):

    matrix_kind = 'numpy'
    tensor_kind = 'jax'
    matrix_type = 'real'
    tensor_type = 'real'

class TestKroneckerSum13(TestKroneckerSum11):

    matrix_kind = 'jax'
    tensor_kind = 'numpy'
    matrix_type = 'real'
    tensor_type = 'real'

class TestKroneckerSum14(TestKroneckerSum11):

    matrix_kind = 'jax'
    tensor_kind = 'jax'
    matrix_type = 'real'
    tensor_type = 'real'

class TestKroneckerSum21(TestKroneckerSum11):

    matrix_kind = 'numpy'
    tensor_kind = 'numpy'
    matrix_type = 'complex'
    tensor_type = 'real'

class TestKroneckerSum22(TestKroneckerSum11):

    matrix_kind = 'numpy'
    tensor_kind = 'jax'
    matrix_type = 'complex'
    tensor_type = 'real'

class TestKroneckerSum23(TestKroneckerSum11):

    matrix_kind = 'jax'
    tensor_kind = 'numpy'
    matrix_type = 'complex'
    tensor_type = 'real'

class TestKroneckerSum24(TestKroneckerSum11):

    matrix_kind = 'jax'
    tensor_kind = 'jax'
    matrix_type = 'complex'
    tensor_type = 'real'

class TestKroneckerSum31(TestKroneckerSum11):

    matrix_kind = 'numpy'
    tensor_kind = 'numpy'
    matrix_type = 'real'
    tensor_type = 'complex'

class TestKroneckerSum32(TestKroneckerSum11):

    matrix_kind = 'numpy'
    tensor_kind = 'jax'
    matrix_type = 'real'
    tensor_type = 'complex'

class TestKroneckerSum33(TestKroneckerSum11):

    matrix_kind = 'jax'
    tensor_kind = 'numpy'
    matrix_type = 'real'
    tensor_type = 'complex'

class TestKroneckerSum34(TestKroneckerSum11):

    matrix_kind = 'jax'
    tensor_kind = 'jax'
    matrix_type = 'complex'
    tensor_type = 'complex'

class TestKroneckerSum41(TestKroneckerSum11):

    matrix_kind = 'numpy'
    tensor_kind = 'numpy'
    matrix_type = 'complex'
    tensor_type = 'complex'

class TestKroneckerSum42(TestKroneckerSum11):

    matrix_kind = 'numpy'
    tensor_kind = 'jax'
    matrix_type = 'complex'
    tensor_type = 'complex'

class TestKroneckerSum43(TestKroneckerSum11):

    matrix_kind = 'jax'
    tensor_kind = 'numpy'
    matrix_type = 'complex'
    tensor_type = 'complex'

class TestKroneckerSum44(TestKroneckerSum11):

    matrix_kind = 'jax'
    tensor_kind = 'jax'
    matrix_type = 'complex'
    tensor_type = 'complex'

    
if __name__ == '__main__':
    unittest.main()