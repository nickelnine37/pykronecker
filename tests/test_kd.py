import unittest
import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from utils import BaseTestCases

from pykronecker import KroneckerDiag
from pykronecker.utils import kronecker_diag_literal


class TestKroneckerDiag11(BaseTestCases.BaseTest):

    powers = [1.0, 3, 2.0]

    def get_operators(self):
        D = self.get_array([shape[0] for shape in self.shapes], self.matrix_type, self.matrix_kind) 
        return KroneckerDiag(D), kronecker_diag_literal(D)


class TestKroneckerDiag12(TestKroneckerDiag11):

    matrix_kind = 'numpy'
    tensor_kind = 'jax'
    matrix_type = 'real'
    tensor_type = 'real'

class TestKroneckerDiag13(TestKroneckerDiag11):

    matrix_kind = 'jax'
    tensor_kind = 'numpy'
    matrix_type = 'real'
    tensor_type = 'real'

class TestKroneckerDiag14(TestKroneckerDiag11):

    matrix_kind = 'jax'
    tensor_kind = 'jax'
    matrix_type = 'real'
    tensor_type = 'real'

class TestKroneckerDiag21(TestKroneckerDiag11):

    matrix_kind = 'numpy'
    tensor_kind = 'numpy'
    matrix_type = 'complex'
    tensor_type = 'real'

class TestKroneckerDiag22(TestKroneckerDiag11):

    matrix_kind = 'numpy'
    tensor_kind = 'jax'
    matrix_type = 'complex'
    tensor_type = 'real'

class TestKroneckerDiag23(TestKroneckerDiag11):

    matrix_kind = 'jax'
    tensor_kind = 'numpy'
    matrix_type = 'complex'
    tensor_type = 'real'

class TestKroneckerDiag24(TestKroneckerDiag11):

    matrix_kind = 'jax'
    tensor_kind = 'jax'
    matrix_type = 'complex'
    tensor_type = 'real'

class TestKroneckerDiag31(TestKroneckerDiag11):

    matrix_kind = 'numpy'
    tensor_kind = 'numpy'
    matrix_type = 'real'
    tensor_type = 'complex'

class TestKroneckerDiag32(TestKroneckerDiag11):

    matrix_kind = 'numpy'
    tensor_kind = 'jax'
    matrix_type = 'real'
    tensor_type = 'complex'

class TestKroneckerDiag33(TestKroneckerDiag11):

    matrix_kind = 'jax'
    tensor_kind = 'numpy'
    matrix_type = 'real'
    tensor_type = 'complex'

class TestKroneckerDiag34(TestKroneckerDiag11):

    matrix_kind = 'jax'
    tensor_kind = 'jax'
    matrix_type = 'complex'
    tensor_type = 'complex'

class TestKroneckerDiag41(TestKroneckerDiag11):

    matrix_kind = 'numpy'
    tensor_kind = 'numpy'
    matrix_type = 'complex'
    tensor_type = 'complex'

class TestKroneckerDiag42(TestKroneckerDiag11):

    matrix_kind = 'numpy'
    tensor_kind = 'jax'
    matrix_type = 'complex'
    tensor_type = 'complex'

class TestKroneckerDiag43(TestKroneckerDiag11):

    matrix_kind = 'jax'
    tensor_kind = 'numpy'
    matrix_type = 'complex'
    tensor_type = 'complex'

class TestKroneckerDiag44(TestKroneckerDiag11):

    matrix_kind = 'jax'
    tensor_kind = 'jax'
    matrix_type = 'complex'
    tensor_type = 'complex'

    
if __name__ == '__main__':
    unittest.main()