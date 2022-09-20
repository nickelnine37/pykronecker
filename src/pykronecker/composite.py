from __future__ import annotations
from abc import ABC

import numpy as np
from numpy import ndarray
from pykronecker.base import KroneckerOperator
from pykronecker.utils import numeric

"""
The classes in this file are used to create composite operators. This is the result of adding or multiplying 
two simpler operators together. These classes never need to be created explicitly, but are implicitly 
created whenever two operators are summed or multiplied. 

E.g.

>>> A = KroneckerProduct(A1, A2, A3)
>>> B = KroneckerSum(B1, B2, B3)

>>> C1 = A + B
>>> assert isinstance(C1, OperatorSum)

>>> C2 = A @ B
>>> assert isinstance(C1, OperatorProduct)

This abstraction can be used indefinitely to create higher and higher order composite operators. 
"""


class CompositeOperator(KroneckerOperator, ABC):
    """
    This is an abstract class grouping together the proceeding two composite operators.
    """

    def __init__(self, A: KroneckerOperator, B: KroneckerOperator):
        """
        Initialise a general composite operator takes two consistent operators A and B
        """

        self.check_operators_consistent(A, B)
        self.A = A
        self.B = B
        self.shape = self.A.shape
        self.tensor_shape = A.tensor_shape
        self.dtype = np.result_type(A.dtype, B.dtype)
    
    def __copy__(self) -> 'CompositeOperator':
        new = self.__class__(self.A.__copy__(), self.B.__copy__())
        new.factor = self.factor
        return new    
    
    def __deepcopy__(self, memodict=None) -> 'CompositeOperator':
        new = self.__class__(self.A.__deepcopy__(), self.B.__deepcopy__())
        new.factor = self.factor
        return new

    def conj(self) -> 'CompositeOperator':
        return np.conj(self.factor) * self.__class__(self.A.conj(), self.B.conj())


class OperatorSum(CompositeOperator):
    """
    Used to represent a chain of Kronecker objects summed together. No need for this class to be
    instantiated by the user. It is used mainly as an internal representation for defining the
    behaviour of composite operators. The internal state of this operator is simply two operators
    A and B.
    """

    def __init__(self, A: KroneckerOperator, B: KroneckerOperator):
        """
        Create an OperatorSum: C = A + B
        """
        super().__init__(A, B)

    def operate(self, other: ndarray) -> ndarray:
        return self.factor * (self.A.operate(other) + self.B.operate(other))

    @property
    def T(self) -> 'OperatorSum':
        return self.factor * OperatorSum(self.A.T, self.B.T)

    def to_array(self) -> ndarray:
        return self.factor * (self.A.to_array() + self.B.to_array())

    def diag(self) -> ndarray:
        return self.factor * (self.A.diag() + self.B.diag())
    
    def __pow__(self, power: numeric, modulo=None) -> KroneckerOperator:
        raise NotImplementedError

    def inv(self) -> KroneckerOperator:
        raise NotImplementedError

    def __repr__(self) -> str:
        return 'OperatorSum({}, {})'.format(self.A.__repr__(), self.B.__repr__())


class OperatorProduct(CompositeOperator):
    """
    Used to represent a chain of Kronecker objects matrix-multiplied together. No need for this class to be
    instantiated by the user. It is used mainly as an internal representation for defining the
    behaviour of composite operators.
    """

    def __pow__(self, power: numeric, modulo=None) -> 'KroneckerOperator':
        raise NotImplementedError

    def __init__(self, A: KroneckerOperator, B: KroneckerOperator):
        """
        Create an OperatorSum: C = A + B
        """
        super().__init__(A, B)

    def operate(self, other: ndarray) -> ndarray:
        return self.factor * (self.A @ (self.B @ other))

    def inv(self) -> 'OperatorProduct':
        return (1 / self.factor) * OperatorProduct(self.B.inv(), self.A.inv())

    @property
    def T(self) -> 'OperatorProduct':
        return self.factor * OperatorProduct(self.B.T, self.A.T)

    def to_array(self) -> ndarray:
        return self.factor * self.A.to_array() @ self.B.to_array()

    def diag(self) -> ndarray:
        return self.factor * (self.A.T * self.B).sum(0)

    def __repr__(self) -> str:
        return 'OperatorProduct({}, {})'.format(self.A.__repr__(), self.B.__repr__())
