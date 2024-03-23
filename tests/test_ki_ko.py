import unittest
import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from utils import BaseTestCases

from pykronecker import KroneckerIdentity, KroneckerOnes
from pykronecker.utils import kronecker_diag_literal, OperatorError
import numpy as np

class TestKroneckerIdentity(BaseTestCases.BaseTest):

    powers = [1.0, 3, 2.0]

    def get_operators(self):
        KI = KroneckerIdentity(tensor_shape=[shape[0] for shape in self.shapes])
        return KI, np.eye(KI.shape[0])


class TestKroneckerOnes1(BaseTestCases.BaseTest):

    def get_operators(self):
        KI = KroneckerOnes(input_shape=[shape[0] for shape in self.shapes])
        return KI, np.ones(KI.shape)

    def test_inv(self):
        self.assertRaises(np.linalg.LinAlgError, self.A_opt.inv)
    

class TestKroneckerOnes2(BaseTestCases.BaseTest):

    shapes = [(4, 2), (2, 3), (1, 4)]

    def get_operators(self):
        KI = KroneckerOnes(input_shape=[shape[1] for shape in self.shapes], output_shape=[shape[0] for shape in self.shapes])
        return KI, np.ones(KI.shape)
    
    def test_diag(self):
        self.assertRaises(OperatorError, self.A_opt.diag)
    
    def test_inv(self):
        self.assertRaises(OperatorError, self.A_opt.inv)


if __name__ == '__main__':
    unittest.main()