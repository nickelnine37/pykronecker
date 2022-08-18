from __future__ import annotations

import numpy as np
from numpy import ndarray
from typing import Union, List
from numpy.linalg import inv

# import kronecker as kron
from pykronecker.base import KroneckerOperator
from pykronecker.types import numeric
from pykronecker.utils import vec, multiply_tensor_product, multiply_tensor_sum, ten, kronecker_product_literal, kronecker_sum_literal, kronecker_diag_literal


class BasicKroneckerOperator(KroneckerOperator):
    """
    This is a superclass for the KroneckerProduct and KroneckerSum classes, which share a some
    functionality. 
    """
    
    def __init__(self, As: List[ndarray]):
        """
        Initialise by passing in a sequence of square arrays as Numpy arrays or spmatrices
        """

        self.check_valid_matrices(As)
        self.As = As
        N = int(np.prod([A.shape[0] for A in As]))
        self.shape = (N, N)

    def __copy__(self) -> 'BasicKroneckerOperator':
        new = self.__class__([A for A in self.As])
        new.factor = self.factor
        return new

    def __deepcopy__(self, *args, **kwargs) -> 'BasicKroneckerOperator':
        new = self.__class__([A.copy() for A in self.As])
        new.factor = self.factor
        return new
    
    @property
    def T(self) -> 'BasicKroneckerOperator':
        return self.factor * self.__class__([A.T for A in self.As])

class KroneckerProduct(BasicKroneckerOperator):
    """
    Used to represent the object (A1 ⊗ A2 ⊗ ... ⊗ AN), that is the Kronecker product of N square matrices.
    """

    def __init__(self, As: List[ndarray]):
        """
        Initialise by passing in a sequence of square arrays as Numpy arrays or spmatrices
        """
        super().__init__(As)

    def __pow__(self, power: numeric, modulo=None) -> 'KroneckerProduct':
        return self.factor ** power * KroneckerProduct([A ** power for A in self.As])

    def __matmul__(self, other: Union[KroneckerOperator, ndarray]) -> Union[KroneckerOperator, ndarray]:

        # in this case, if other is another Kronecker product, we can get a simpler representation
        if isinstance(other, KroneckerProduct):

            self.check_operators_consistent(self, other)
            return self.factor * other.factor * KroneckerProduct([A1 @ A2 for A1, A2 in zip(self.As, other.As)])

        # otherwise default to creating an OperatorChain
        else:
            return super().__matmul__(other)

    def __mul__(self, other: Union['KroneckerProduct', numeric]) -> KroneckerOperator:

        # kronecker products can be hadamarded against other kronecker products only
        if isinstance(other, KroneckerProduct):

            self.check_operators_consistent(self, other)
            return self.factor * other.factor * KroneckerProduct([A1 * A2 for A1, A2 in zip(self.As, other.As)])

        # otherwise other should be a scalar, handled in the base class
        else:
            return super().__mul__(other)

    def operate(self, other: ndarray) -> ndarray:

        other = np.squeeze(other)

        # handle when other is a vector
        if other.ndim == 1:
            other_ten = ten(other, shape=tuple(A.shape[0] for A in reversed(self.As)))
            return self.factor * vec(multiply_tensor_product(self.As, other_ten))

        # handle when other is a matrix of column vectors
        elif other.ndim == 2 and other.shape[0] == len(self):

            out = np.zeros_like(other)

            for i in range(other.shape[1]):
                other_ten = ten(other[:, i], shape=tuple(A.shape[0] for A in reversed(self.As)))
                out[:, i] = vec(multiply_tensor_product(self.As, other_ten))

            return self.factor * out

        # handle when other is a tensor
        else:
            return self.factor * multiply_tensor_product(self.As, other)

    def inv(self) -> 'KroneckerProduct':
        return (1 / self.factor) * KroneckerProduct([inv(A) for A in self.As])

    def to_array(self) -> ndarray:
        return self.factor * kronecker_product_literal(self.As)

    def __repr__(self) -> str:
        return 'KroneckerProduct({})'.format(' ⊗ '.join([str(len(A)) for A in self.As]))

    def __str__(self) -> str:
        return 'KroneckerProduct({})'.format(' ⊗ '.join([str(len(A)) for A in self.As]))


class KroneckerSum(BasicKroneckerOperator):
    """
    Used to represent the object (A1 ⊕ A2 ⊕ ... ⊕ AN), that is the Kronecker sum of N square matrices.
    """

    def __init__(self, As: List[ndarray]):
        """
        Initialise by passing in a sequence of square arrays as Numpy arrays or spmatrices
        """
        super().__init__(As)

    def __pow__(self, power: numeric, modulo=None) -> 'KroneckerOperator':
        raise NotImplementedError

    def operate(self, other: ndarray) -> ndarray:

        other = np.squeeze(other)

        # handle when other is a vector
        if other.ndim == 1:
            other_ten = ten(other, shape=tuple(A.shape[0] for A in reversed(self.As)))
            return self.factor * vec(multiply_tensor_sum(self.As, other_ten))

        # handle when other is a matrix of column vectors
        elif other.ndim == 2 and other.shape[0] == len(self):

            out = np.zeros_like(other)

            for i in range(other.shape[1]):
                other_ten = ten(other[:, i], shape=tuple(A.shape[0] for A in reversed(self.As)))
                out[:, i] = vec(multiply_tensor_sum(self.As, other_ten))

            return self.factor * out

        # handle when other is a tensor
        else:
            return self.factor * multiply_tensor_sum(self.As, other)

    def inv(self) -> KroneckerOperator:
        raise NotImplementedError

    def to_array(self) -> ndarray:
        return self.factor * kronecker_sum_literal(self.As)

    def __repr__(self) -> str:
        return 'KroneckerSum({})'.format(' ⊗ '.join([str(len(A)) for A in self.As]))

    def __str__(self) -> str:
        return 'KroneckerSum({})'.format(' ⊗ '.join([str(len(A)) for A in self.As]))


class KroneckerDiag(KroneckerOperator):
    """
    Used to represent a general diagonal matrix of size N1 x N2 x ... x NN
    """

    def __init__(self, A: ndarray):
        """
        Initialise with a tensor A of shape (Nn, ..., N1)
        """

        assert isinstance(A, ndarray)
        assert A.ndim > 1, 'The operator diagonal A should be in tensor format, but it is in vector format'

        self.A = A
        N = int(np.prod(A.shape))
        self.shape = (N, N)

    def __copy__(self) -> 'KroneckerDiag':
        new = KroneckerDiag(self.A)
        new.factor = self.factor
        return new

    def __deepcopy__(self, memodict={}) -> 'KroneckerDiag':
        new = KroneckerDiag(self.A.copy())
        new.factor = self.factor
        return new

    def __pow__(self, power: numeric, modulo=None) -> 'KroneckerDiag':

        if power < 0:
            raise NotImplementedError

        new = KroneckerDiag(self.A ** power)
        new.factor = self.factor ** power
        return new

    def __matmul__(self, other: Union[KroneckerOperator, ndarray]) -> Union[KroneckerOperator, ndarray]:

        # in this case, if other is another KroneckerDiag, we can get a simpler representation
        if isinstance(other, KroneckerDiag):

            self.check_operators_consistent(self, other)

            return self.factor * other.factor * KroneckerDiag(self.A * other.A)

        else:
            return super().__matmul__(other)

    def __mul__(self, other: Union['KroneckerDiag', numeric]) -> KroneckerOperator:

        # kronecker diags can be hadamarded against other kronecker diags only
        if isinstance(other, KroneckerDiag):

            self.check_operators_consistent(self, other)
            return self.factor * other.factor * KroneckerDiag(self.A * other.A)

        # otherwise other should be a scalar, handled in the base class
        else:
            return super().__mul__(other)

    def operate(self, other: ndarray) -> ndarray:

        # handle when other is a vector
        if other.ndim == 1:
            return self.factor * vec(self.A) * other

        # handle when other is a matrix of column vectors
        elif other.ndim == 2 and other.shape[0] == len(self):

            out = np.zeros_like(other)

            for i in range(other.shape[1]):
                out[:, i] = vec(self.A) * other[:, i]

            return self.factor * out

        # handle when other is a tensor
        else:
            return self.factor * self.A * other

    def inv(self) -> 'KroneckerDiag':
        return self.factor * KroneckerDiag(1 / self.A)

    @property
    def T(self) -> 'KroneckerDiag':
        return self

    def to_array(self) -> ndarray:
        return self.factor * kronecker_diag_literal(self.A)

    def __repr__(self) -> str:
        return 'KroneckerDiag({})'.format(' ⊗ '.join([str(i) for i in reversed(self.A.shape)]))

    def __str__(self) -> str:
        return 'KroneckerDiag({})'.format(' ⊗ '.join([str(i) for i in reversed(self.A.shape)]))


