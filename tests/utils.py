from __future__ import annotations
from numpy import ndarray
import numpy as np

from pykronecker.base import KroneckerOperator
from pykronecker import KroneckerProduct, KroneckerDiag, KroneckerSum
from pykronecker.operators import KroneckerIdentity, KroneckerOnes
from pykronecker.utils import kronecker_product_literal, kronecker_sum_literal, kronecker_diag_literal, vec, ten

import jax
import jax.numpy as jnp

import unittest
from itertools import product

from typing import Literal, Optional, Tuple
# from jax import config
# config.update("jax_enable_x64", True)

class BaseTestCases:

    class BaseTest(unittest.TestCase):

        matrix_kind = 'numpy'
        tensor_kind = 'numpy'
        matrix_type = 'real'
        tensor_type = 'real'
        shapes = [(2, 2), (3, 3), (4, 4)]

        A_opt: Optional[KroneckerOperator] = None
        A_lit: Optional[np.ndarray | jnp.ndarray] = None
        X_l: Optional[np.ndarray | jnp.ndarray] = None
        X_r: Optional[np.ndarray | jnp.ndarray] = None
        x_l: Optional[np.ndarray | jnp.ndarray] = None
        x_r: Optional[np.ndarray | jnp.ndarray] = None
        P_l: Optional[np.ndarray | jnp.ndarray] = None
        P_r: Optional[np.ndarray | jnp.ndarray] = None

        powers = [1.0, 3, 2.0, -1]

        def setUp(self):
            np.random.seed(0)

            self.A_opt, self.A_lit = self.get_operators()
            self.X_l, self.X_r = self.get_tensors()
            self.x_l, self.x_r = self.get_vectors()
            self.P_l, self.P_r = self.get_multivecs()

        @staticmethod
        def get_array(shape: tuple, dtype: str, kind: str):

            A = np.random.random(size=shape).astype(np.complex128 if dtype == 'complex' else np.float32)
            if dtype == 'complex':
                A += 1j * np.random.random(size=shape).astype(np.complex128 if dtype == 'complex' else np.float32)

            if kind == 'jax':
                return jnp.asarray(A)

            return A

        def get_operators(self) -> Tuple[KroneckerOperator, np.ndarray]:
            """ Return a KroneckerOperator and the equivalent matrix version """
            raise NotImplementedError('Should be implemented by subclass')

        def get_tensors(self):
            return (self.get_array(self.A_opt.output_shape, self.tensor_type, self.tensor_kind), 
                    self.get_array(self.A_opt.input_shape, self.tensor_type, self.tensor_kind))

        def get_vectors(self):
            return (self.get_array(self.A_opt.shape[0], self.tensor_type, self.tensor_kind), 
                    self.get_array(self.A_opt.shape[1], self.tensor_type, self.tensor_kind))

        def get_multivecs(self):
            return (np.concatenate([self.get_array(self.A_opt.shape[0], self.tensor_type, self.tensor_kind)[None, :] for i in range(3)], axis=0), 
                    np.concatenate([self.get_array(self.A_opt.shape[1], self.tensor_type, self.tensor_kind)[:, None] for i in range(3)], axis=1))

        def assertClose(self, a, b):
            self.assertTrue(np.allclose(a, b, rtol=1e-4, atol=1e-4))

        def test_conversions(self):
            """ Test basic conversions """
            self.assertClose(self.A_lit, self.A_opt.to_array())
            self.assertClose(self.A_lit.T, self.A_opt.T.to_array())
            self.assertClose(self.A_lit.conj(), self.A_opt.conj().to_array())
            self.assertClose(self.A_lit.conj().T, self.A_opt.H.to_array())
            self.assertClose(self.A_lit, (+self.A_opt).to_array())
            self.assertClose(-self.A_lit, (-self.A_opt).to_array())

        def test_indexing(self):
            """
            Test all supported indexing functions work
            """

            self.assertClose(self.A_opt[2], self.A_lit[2])
            self.assertClose(self.A_opt[(2,)], self.A_lit[(2,)])
            self.assertClose(self.A_opt[2:5], self.A_lit[2:5])
            self.assertClose(self.A_opt[2:8:3], self.A_lit[2:8:3])
            self.assertClose(self.A_opt[:, 2], self.A_lit[:, 2])
            self.assertClose(self.A_opt[:, 2:5], self.A_lit[:, 2:5])
            self.assertClose(self.A_opt[:, 2:8:3], self.A_lit[:, 2:8:3])
            self.assertClose(self.A_opt[2, :], self.A_lit[2, :])
            self.assertClose(self.A_opt[2:5, :], self.A_lit[2:5, :])
            self.assertClose(self.A_opt[2:8:3, :], self.A_lit[2:8:3, :])
            self.assertClose(self.A_opt[2:5, 2:8:3], self.A_lit[2:5, 2:8:3])
            self.assertClose(self.A_opt[2, 5], self.A_lit[2, 5])
            self.assertClose(self.A_opt[:], self.A_lit[:])
            self.assertClose(self.A_opt[:, :], self.A_lit[:, :])
            self.assertRaises(IndexError, lambda: self.A_opt[2, 3, 4])
            self.assertRaises(IndexError, lambda: self.A_opt['hey'])
            self.assertRaises(IndexError, lambda: self.A_opt['a', 'b'])

        def test_multiply(self):
            """ Test forwards and backwards multiplication """

            self.assertClose(self.A_lit @ self.x_r, self.A_opt @ self.x_r)
            self.assertClose(self.A_lit @ vec(self.X_r), vec(self.A_opt @ self.X_r))
            self.assertClose(self.A_lit @ self.P_r, self.A_opt @ self.P_r)
            self.assertClose(np.matmul(self.A_lit, self.x_r), np.matmul(self.A_opt, self.x_r))

            self.assertClose(self.x_l @ self.A_lit, self.x_l @ self.A_opt)
            self.assertClose(vec(self.X_l) @ self.A_lit, vec(self.X_l @ self.A_opt))
            self.assertClose(self.P_l @ self.A_lit, self.P_l @ self.A_opt)
            self.assertClose(np.matmul(self.x_l, self.A_lit), np.matmul(self.x_l, self.A_opt))

            if self.A_opt.shape[0] == self.A_opt.shape[1]:
                self.assertClose(self.x_l @ self.A_lit @ self.x_r, self.x_l @ self.A_opt @ self.x_r)
                self.assertClose(self.P_l @ self.A_lit @ self.P_r, self.P_l @ self.A_opt @ self.P_r)
                
        def test_sum(self):
            """ Test summing operations """
            self.assertClose(self.A_lit.sum(0), self.A_opt.sum(0))
            self.assertClose(self.A_lit.sum(1), self.A_opt.sum(1))
            self.assertClose(self.A_lit.sum(), self.A_opt.sum())
            self.assertClose(self.A_lit.sum(-1), self.A_opt.sum(-1))
            self.assertRaises(ValueError, lambda: self.A_opt.sum(2))
        
        def test_scalar_multiply(self):
            """ Test scalar multiply operations with ints, floats, np.floats and complex numbers """
            for factor in [-1, np.random.normal(), np.float64(np.random.normal()), np.random.normal() + 1j * np.random.normal()]:
                self.assertClose(factor * self.A_lit, (factor * self.A_opt).to_array())
                self.assertClose(self.A_lit * factor, (self.A_opt * factor).to_array())
                self.assertClose(np.multiply(factor, self.A_lit), np.multiply(factor, self.A_opt).to_array())
                self.assertClose(np.multiply(self.A_lit, factor), np.multiply(self.A_opt, factor).to_array())
                self.assertClose(self.A_lit / factor, (self.A_opt / factor).to_array())

            self.assertRaises(TypeError, lambda: self.A_opt * 'hey')

        def test_diag(self):
            """ Test diag operation """
            self.assertClose(np.diag(self.A_lit), self.A_opt.diag())

        def test_to_string(self):
            """ Test string operations """
            self.A_opt.__str__()
            self.A_opt.__repr__()

        def test_copy(self):
            """ Test copying works """
            self.assertClose(self.A_lit, self.A_opt.copy().to_array())
            self.assertClose(self.A_lit, self.A_opt.deepcopy().to_array())

        def test_exceptions(self):

            self.assertRaises(NotImplementedError, lambda: np.divide(self.A_opt, 5))
            self.assertRaises(TypeError, lambda: self.A_opt @ 5)
            self.assertRaises(TypeError, lambda: 5 @ self.A_opt)

        def test_pow(self):

            for power in self.powers:
                self.assertClose(self.A_lit ** power, (self.A_opt ** power).to_array())

        def assert_hadamard(self):
            self.assertClose(self.A_lit * self.A_lit (self.A_opt * self.A_opt).to_array())
            B_opt, B_lit = self.get_operators()
            self.assertClose(self.A_lit * B_lit (self.A_opt * B_opt).to_array())

        def test_inv(self):
            self.assertClose(np.eye(self.A_opt.shape[0]), (self.A_opt.inv() @ self.A_opt).to_array())

