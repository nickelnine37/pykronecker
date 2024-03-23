import unittest
import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from utils import BaseTestCases

from pykronecker import KroneckerProduct, KroneckerSum, KroneckerDiag, KroneckerIdentity, KroneckerOnes
from pykronecker.utils import kronecker_product_literal, OperatorError, kronecker_sum_literal, kronecker_diag_literal
import numpy as np

class TestSum1(BaseTestCases.BaseTest):

    def get_operators(self):

        As = [self.get_array(shape, self.matrix_type, self.matrix_kind) for shape in self.shapes]
        Bs = [self.get_array(shape, self.matrix_type, self.matrix_kind) for shape in self.shapes]

        KP = KroneckerProduct(As)
        KS = KroneckerSum(Bs)

        return KP + KS, kronecker_product_literal(As) + kronecker_sum_literal(Bs)
    
    def test_inv(self):
        self.assertRaises(NotImplementedError, self.A_opt.inv)

    def test_pow(self):
        self.assertRaises(NotImplementedError, lambda: self.A_opt ** 2)


class TestProduct1(BaseTestCases.BaseTest):

    def get_operators(self):

        As = [self.get_array(shape, self.matrix_type, self.matrix_kind) for shape in self.shapes]
        Bs = [self.get_array(shape, self.matrix_type, self.matrix_kind) for shape in self.shapes]

        KP = KroneckerProduct(As)
        KS = KroneckerSum(Bs)

        return KP @ KS, kronecker_product_literal(As) @ kronecker_sum_literal(Bs)
    
    def test_inv(self):
        self.assertRaises(NotImplementedError, self.A_opt.inv)

    def test_pow(self):
        self.assertRaises(NotImplementedError, lambda: self.A_opt ** 2)


class TestProduct2(BaseTestCases.BaseTest):

    def get_operators(self):

        As = [self.get_array(shape, self.matrix_type, self.matrix_kind) for shape in self.shapes]
        Bs = [self.get_array(shape, self.matrix_type, self.matrix_kind) for shape in self.shapes]

        KP1 = KroneckerProduct(As)
        KP2 = KroneckerProduct(Bs)

        return KP1 @ KP2, kronecker_product_literal(As) @ kronecker_product_literal(Bs)
    

class TestComposite1(BaseTestCases.BaseTest):

    def get_operators(self):

        As = [self.get_array(shape, self.matrix_type, self.matrix_kind) for shape in self.shapes]
        Bs = [self.get_array(shape, self.matrix_type, self.matrix_kind) for shape in self.shapes]
        D = self.get_array([shape[0] for shape in self.shapes], self.matrix_type, self.matrix_kind) 

        KP = KroneckerProduct(As)
        KS = KroneckerSum(Bs)
        KD = KroneckerDiag(D)

        return KP @ KS + KD, kronecker_product_literal(As) @ kronecker_sum_literal(Bs) + kronecker_diag_literal(D)
    
    def test_inv(self):
        self.assertRaises(NotImplementedError, self.A_opt.inv)

    def test_pow(self):
        self.assertRaises(NotImplementedError, lambda: self.A_opt ** 2)


class TestComposite2(BaseTestCases.BaseTest):

    def get_operators(self):

        As = [self.get_array(shape, self.matrix_type, self.matrix_kind) for shape in self.shapes]
        Bs = [self.get_array(shape, self.matrix_type, self.matrix_kind) for shape in self.shapes]
        D = self.get_array([shape[0] for shape in self.shapes], self.matrix_type, self.matrix_kind) 

        KP = KroneckerProduct(As)
        KS = KroneckerSum(Bs)
        KD = KroneckerDiag(D)

        return (KP + KS) @ (KD - KP), (kronecker_product_literal(As) + kronecker_sum_literal(Bs)) @ (kronecker_diag_literal(D) - kronecker_product_literal(As))
    
    def test_inv(self):
        self.assertRaises(NotImplementedError, self.A_opt.inv)

    def test_pow(self):
        self.assertRaises(NotImplementedError, lambda: self.A_opt ** 2)

    def test_diag(self):
        self.assertRaises(TypeError, lambda: self.A_opt.diag())



class TestComposite3(BaseTestCases.BaseTest):

    def get_operators(self):

        As = [self.get_array(shape, self.matrix_type, self.matrix_kind) for shape in self.shapes]

        KP = KroneckerProduct(As)
        KI = KroneckerIdentity(like=KP)

        return KP + KI, kronecker_product_literal(As) + np.eye(KP.shape[0])
    
    def test_inv(self):
        self.assertRaises(NotImplementedError, self.A_opt.inv)

    def test_pow(self):
        self.assertRaises(NotImplementedError, lambda: self.A_opt ** 2)


class TestComposite4(BaseTestCases.BaseTest):

    def get_operators(self):

        As = [self.get_array(shape, self.matrix_type, self.matrix_kind) for shape in self.shapes]

        KP = KroneckerProduct(As)
        KI = KroneckerIdentity(like=KP)

        return KP @ KI, kronecker_product_literal(As)


class TestComposite5(BaseTestCases.BaseTest):

    def get_operators(self):
        As = [self.get_array(shape, self.matrix_type, self.matrix_kind) for shape in self.shapes]
        KP = KroneckerProduct(As)
        return KP + 1, kronecker_product_literal(As) + 1
    
    def test_inv(self):
        self.assertRaises(NotImplementedError, self.A_opt.inv)

    def test_pow(self):
        self.assertRaises(NotImplementedError, lambda: self.A_opt ** 2)


class TestComposite6(BaseTestCases.BaseTest):

    shapes = [(4, 2), (2, 3), (1, 4)]

    def get_operators(self):
        As = [self.get_array(shape, self.matrix_type, self.matrix_kind) for shape in self.shapes]
        return 1 + KroneckerProduct(As), 1 + kronecker_product_literal(As)

    def test_diag(self):
        self.assertRaises(OperatorError, self.A_opt.diag)
    
    def test_inv(self):
        self.assertRaises(NotImplementedError, self.A_opt.inv)
    
    def test_pow(self):
        self.assertRaises(NotImplementedError, lambda: self.A_opt ** 2)


if __name__ == '__main__':
    unittest.main()


