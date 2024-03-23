import unittest
import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from utils import BaseTestCases

from pykronecker import KroneckerProduct
from pykronecker.utils import kronecker_product_literal, OperatorError


class TestKroneckerProductSquare11(BaseTestCases.BaseTest):

    def get_operators(self):
        As = [self.get_array(shape, self.matrix_type, self.matrix_kind) for shape in self.shapes]
        return KroneckerProduct(As), kronecker_product_literal(As)


class TestKroneckerProductSquare12(TestKroneckerProductSquare11):

    matrix_kind = 'numpy'
    tensor_kind = 'jax'
    matrix_type = 'real'
    tensor_type = 'real'

class TestKroneckerProductSquare13(TestKroneckerProductSquare11):

    matrix_kind = 'jax'
    tensor_kind = 'numpy'
    matrix_type = 'real'
    tensor_type = 'real'

class TestKroneckerProductSquare14(TestKroneckerProductSquare11):

    matrix_kind = 'jax'
    tensor_kind = 'jax'
    matrix_type = 'real'
    tensor_type = 'real'

class TestKroneckerProductSquare21(TestKroneckerProductSquare11):

    matrix_kind = 'numpy'
    tensor_kind = 'numpy'
    matrix_type = 'complex'
    tensor_type = 'real'

class TestKroneckerProductSquare22(TestKroneckerProductSquare11):

    matrix_kind = 'numpy'
    tensor_kind = 'jax'
    matrix_type = 'complex'
    tensor_type = 'real'

class TestKroneckerProductSquare23(TestKroneckerProductSquare11):

    matrix_kind = 'jax'
    tensor_kind = 'numpy'
    matrix_type = 'complex'
    tensor_type = 'real'

class TestKroneckerProductSquare24(TestKroneckerProductSquare11):

    matrix_kind = 'jax'
    tensor_kind = 'jax'
    matrix_type = 'complex'
    tensor_type = 'real'

class TestKroneckerProductSquare31(TestKroneckerProductSquare11):

    matrix_kind = 'numpy'
    tensor_kind = 'numpy'
    matrix_type = 'real'
    tensor_type = 'complex'

class TestKroneckerProductSquare32(TestKroneckerProductSquare11):

    matrix_kind = 'numpy'
    tensor_kind = 'jax'
    matrix_type = 'real'
    tensor_type = 'complex'

class TestKroneckerProductSquare33(TestKroneckerProductSquare11):

    matrix_kind = 'jax'
    tensor_kind = 'numpy'
    matrix_type = 'real'
    tensor_type = 'complex'

class TestKroneckerProductSquare34(TestKroneckerProductSquare11):

    matrix_kind = 'jax'
    tensor_kind = 'jax'
    matrix_type = 'complex'
    tensor_type = 'complex'

class TestKroneckerProductSquare41(TestKroneckerProductSquare11):

    matrix_kind = 'numpy'
    tensor_kind = 'numpy'
    matrix_type = 'complex'
    tensor_type = 'complex'

class TestKroneckerProductSquare42(TestKroneckerProductSquare11):

    matrix_kind = 'numpy'
    tensor_kind = 'jax'
    matrix_type = 'complex'
    tensor_type = 'complex'

class TestKroneckerProductSquare43(TestKroneckerProductSquare11):

    matrix_kind = 'jax'
    tensor_kind = 'numpy'
    matrix_type = 'complex'
    tensor_type = 'complex'

class TestKroneckerProductSquare44(TestKroneckerProductSquare11):

    matrix_kind = 'jax'
    tensor_kind = 'jax'
    matrix_type = 'complex'
    tensor_type = 'complex'

    
if __name__ == '__main__':
    unittest.main()