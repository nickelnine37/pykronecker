import unittest
import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from utils import BaseTestCases

from pykronecker import KroneckerProduct
from pykronecker.utils import kronecker_product_literal, OperatorError


class TestKroneckerProductNonSquare11(BaseTestCases.BaseTest):

    shapes = [(4, 2), (2, 3), (1, 4)]

    def get_operators(self):
        As = [self.get_array(shape, self.matrix_type, self.matrix_kind) for shape in self.shapes]
        return KroneckerProduct(As), kronecker_product_literal(As)

    def test_diag(self):
        self.assertRaises(OperatorError, self.A_opt.diag)
    
    def test_inv(self):
        self.assertRaises(OperatorError, self.A_opt.inv)


class TestKroneckerProductNonSquare12(TestKroneckerProductNonSquare11):

    matrix_kind = 'numpy'
    tensor_kind = 'jax'
    matrix_type = 'real'
    tensor_type = 'real'

class TestKroneckerProductNonSquare13(TestKroneckerProductNonSquare11):

    matrix_kind = 'jax'
    tensor_kind = 'numpy'
    matrix_type = 'real'
    tensor_type = 'real'

class TestKroneckerProductNonSquare14(TestKroneckerProductNonSquare11):

    matrix_kind = 'jax'
    tensor_kind = 'jax'
    matrix_type = 'real'
    tensor_type = 'real'

class TestKroneckerProductNonSquare21(TestKroneckerProductNonSquare11):

    matrix_kind = 'numpy'
    tensor_kind = 'numpy'
    matrix_type = 'complex'
    tensor_type = 'real'

class TestKroneckerProductNonSquare22(TestKroneckerProductNonSquare11):

    matrix_kind = 'numpy'
    tensor_kind = 'jax'
    matrix_type = 'complex'
    tensor_type = 'real'

class TestKroneckerProductNonSquare23(TestKroneckerProductNonSquare11):

    matrix_kind = 'jax'
    tensor_kind = 'numpy'
    matrix_type = 'complex'
    tensor_type = 'real'

class TestKroneckerProductNonSquare24(TestKroneckerProductNonSquare11):

    matrix_kind = 'jax'
    tensor_kind = 'jax'
    matrix_type = 'complex'
    tensor_type = 'real'

class TestKroneckerProductNonSquare31(TestKroneckerProductNonSquare11):

    matrix_kind = 'numpy'
    tensor_kind = 'numpy'
    matrix_type = 'real'
    tensor_type = 'complex'

class TestKroneckerProductNonSquare32(TestKroneckerProductNonSquare11):

    matrix_kind = 'numpy'
    tensor_kind = 'jax'
    matrix_type = 'real'
    tensor_type = 'complex'

class TestKroneckerProductNonSquare33(TestKroneckerProductNonSquare11):

    matrix_kind = 'jax'
    tensor_kind = 'numpy'
    matrix_type = 'real'
    tensor_type = 'complex'

class TestKroneckerProductNonSquare34(TestKroneckerProductNonSquare11):

    matrix_kind = 'jax'
    tensor_kind = 'jax'
    matrix_type = 'complex'
    tensor_type = 'complex'

class TestKroneckerProductNonSquare41(TestKroneckerProductNonSquare11):

    matrix_kind = 'numpy'
    tensor_kind = 'numpy'
    matrix_type = 'complex'
    tensor_type = 'complex'

class TestKroneckerProductNonSquare42(TestKroneckerProductNonSquare11):

    matrix_kind = 'numpy'
    tensor_kind = 'jax'
    matrix_type = 'complex'
    tensor_type = 'complex'

class TestKroneckerProductNonSquare43(TestKroneckerProductNonSquare11):

    matrix_kind = 'jax'
    tensor_kind = 'numpy'
    matrix_type = 'complex'
    tensor_type = 'complex'

class TestKroneckerProductNonSquare44(TestKroneckerProductNonSquare11):

    matrix_kind = 'jax'
    tensor_kind = 'jax'
    matrix_type = 'complex'
    tensor_type = 'complex'

    
if __name__ == '__main__':
    unittest.main()