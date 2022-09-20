from __future__ import annotations

from abc import ABC
from functools import reduce
from itertools import combinations

import numpy as np
from numpy import ndarray
from typing import Union, List
from numpy.linalg import inv

from pykronecker.base import KroneckerOperator
from pykronecker.composite import OperatorSum, OperatorProduct
from pykronecker.utils import numeric
from pykronecker.utils import multiply_tensor_product, multiply_tensor_sum, multiply_tensor_diag, multiply_tensor_identity, kronecker_product_literal, kronecker_sum_literal, kronecker_diag_literal, \
    vec, ten


class BasicKroneckerOperator(KroneckerOperator, ABC):
    """
    This is a superclass for the KroneckerProduct and KroneckerSum classes, which share a some
    functionality. 
    """
    
    def __init__(self, As: List[ndarray]):
        """
        Initialise by passing in a sequence of square arrays as Numpy arrays
        """

        self.check_valid_matrices(As)
        self.As = As
        self.tensor_shape = tuple(A.shape[0] for A in As)
        N = int(np.prod(self.tensor_shape))
        self.shape = (N, N)
        self.dtype = np.result_type(*self.As)

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

    def conj(self) -> 'BasicKroneckerOperator':
        return np.conj(self.factor) * self.__class__([A.conj() for A in self.As])


class KroneckerProduct(BasicKroneckerOperator):
    """
    Used to represent the object (A1 ⊗ A2 ⊗ ... ⊗ AN), that is the Kronecker product of N square matrices.
    """

    def __init__(self, As: List[ndarray]):
        """
        Initialise by passing in a sequence of square arrays as Numpy arrays
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

    def __mul__(self, other: Union['KroneckerOperator', numeric]) -> KroneckerOperator:

        if isinstance(other, KroneckerOperator):

            self.check_operators_consistent(self, other)

            if isinstance(other, KroneckerProduct):
                return self.factor * other.factor * KroneckerProduct([A1 * A2 for A1, A2 in zip(self.As, other.As)])

            elif isinstance(other, KroneckerDiag):
                return KroneckerDiag(ten(self.diag() * other.diag(), like=other.A))

            elif isinstance(other, KroneckerSum):
                n = len(self.As)
                return self.factor * other.factor * reduce(OperatorSum, [KroneckerProduct([self.As[i] * other.As[i] if i == j else np.diag(np.diag(self.As[j])) for j in range(n)]) for i in range(n)])

            elif isinstance(other, KroneckerIdentity):
                return other.factor * KroneckerDiag(ten(self.diag(), shape=self.tensor_shape))

            elif isinstance(other, OperatorSum):
                return other.factor * OperatorSum(self * other.A, self * other.B)

            else:
                raise TypeError('A KroneckerProduct cannot be multiplied element-wise onto an OperatorProduct')

        # otherwise other should be a scalar, handled in the base class
        else:
            return super().__mul__(other)

    def operate(self, other: ndarray) -> ndarray:
        return self.factor * multiply_tensor_product(self.As, other)

    def inv(self) -> 'KroneckerProduct':
        return (1 / self.factor) * KroneckerProduct([inv(A) for A in self.As])

    def diag(self) -> ndarray:
        return self.factor * vec(np.prod(np.array([np.expand_dims(np.diag(A), axis=[j for j in range(len(self.As)) if j != i]) for i, A in enumerate(self.As)], dtype='object')))

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
        Initialise by passing in a sequence of square arrays as Numpy arrays
        """
        super().__init__(As)

    def __mul__(self, other: Union['KroneckerOperator', numeric]) -> KroneckerOperator:

        if isinstance(other, KroneckerOperator):

            self.check_operators_consistent(self, other)

            if isinstance(other, KroneckerProduct):
                n = len(self.As)
                return self.factor * other.factor * reduce(OperatorSum, [KroneckerProduct([self.As[i] * other.As[i] if i == j else np.diag(np.diag(other.As[j])) for j in range(n)]) for i in range(n)])

            elif isinstance(other, KroneckerDiag):
                return KroneckerDiag(ten(self.diag() * other.diag(), like=other.A))

            # this is a little complex... but it works I promise.
            elif isinstance(other, KroneckerSum):

                n = len(self.As)
                diags = {'A': [np.diag(A) for A in self.As], 'B': [np.diag(A) for A in other.As]}
                diagonal = np.zeros(len(self), dtype=np.result_type(*self.As, *other.As))

                for ind1, ind2 in combinations(range(n), 2):
                    diagonal += reduce(np.kron, [diags['A'][ind1] if i == ind1 else diags['B'][ind2] if i == ind2 else np.ones(len(self.As[i])) for i in range(n)])
                    diagonal += reduce(np.kron, [diags['B'][ind1] if i == ind1 else diags['A'][ind2] if i == ind2 else np.ones(len(self.As[i])) for i in range(n)])

                return self.factor * other.factor * (KroneckerSum([A * B for A, B in zip(self.As, other.As)]) + KroneckerDiag(ten(diagonal, shape=tuple(A.shape[0] for A in self.As))))

            elif isinstance(other, KroneckerIdentity):
                return other.factor * KroneckerDiag(ten(self.diag(), shape=self.tensor_shape))

            elif isinstance(other, OperatorSum):
                return other.factor * OperatorSum(self * other.A, self * other.B)

            else:
                raise TypeError('A KroneckerSum cannot be multiplied element-wise onto an OperatorProduct')

        # otherwise other should be a scalar, handled in the base class
        else:
            return super().__mul__(other)

    def __pow__(self, power: numeric, modulo=None) -> 'KroneckerOperator':
        raise NotImplementedError

    def operate(self, other: ndarray) -> ndarray:
        return self.factor * multiply_tensor_sum(self.As, other)

    def inv(self) -> KroneckerOperator:
        raise NotImplementedError

    def to_array(self) -> ndarray:
        return self.factor * kronecker_sum_literal(self.As)

    def diag(self) -> ndarray:
        return self.factor * vec(np.sum(np.array([np.expand_dims(np.diag(A), axis=[j for j in range(len(self.As)) if j != i]) for i, A in enumerate(self.As)], dtype='object')))

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
        self.tensor_shape = A.shape
        N = int(np.prod(self.tensor_shape))
        self.shape = (N, N)
        self.dtype = A.dtype

    def __mul__(self, other: Union['KroneckerOperator', numeric]) -> KroneckerOperator:

        if isinstance(other, KroneckerOperator):

            self.check_operators_consistent(self, other)

            if isinstance(other, (KroneckerProduct, KroneckerSum, KroneckerDiag, OperatorSum, OperatorProduct)):
                return KroneckerDiag(ten(self.diag() * other.diag(), like=self.A))

            elif isinstance(other, KroneckerIdentity):
                return other.factor * KroneckerDiag(ten(self.diag(), like=self.A))

            else:
                raise TypeError('A KroneckerDiag cannot be multiplied element-wise onto this operator')

        # otherwise other should be a scalar, handled in the base class
        else:
            return super().__mul__(other)

    def __copy__(self) -> 'KroneckerDiag':
        new = KroneckerDiag(self.A)
        new.factor = self.factor
        return new

    def __deepcopy__(self, memodict=None) -> 'KroneckerDiag':
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

    def operate(self, other: ndarray) -> ndarray:
        return self.factor * multiply_tensor_diag(self.A, other)

    def inv(self) -> 'KroneckerDiag':
        return (self.factor ** -1) * KroneckerDiag(1 / self.A)

    @property
    def T(self) -> 'KroneckerDiag':
        return self

    def conj(self) -> 'KroneckerDiag':
        return np.conj(self.factor) * KroneckerDiag(self.A.conj())

    def diag(self) -> ndarray:
        return self.factor * vec(self.A)

    def to_array(self) -> ndarray:
        return self.factor * kronecker_diag_literal(self.A)

    def __repr__(self) -> str:
        return 'KroneckerDiag({})'.format(' ⊗ '.join([str(i) for i in reversed(self.A.shape)]))

    def __str__(self) -> str:
        return 'KroneckerDiag({})'.format(' ⊗ '.join([str(i) for i in reversed(self.A.shape)]))


class KroneckerIdentity(KroneckerOperator):

    def __init__(self, tensor_shape: tuple=None, like: KroneckerOperator=None):
        """
        Initialise an Identity matrix using tensor_shape parameter, or alternatively pass another operator of the same size.
        """

        if tensor_shape is None and like is None:
            raise ValueError('Either shape or like must be passed')

        if tensor_shape is not None:

            if isinstance(tensor_shape, (tuple, list, ndarray)):
                self.tensor_shape = tensor_shape
                N = int(np.prod(tensor_shape))
                self.shape = (N, N)

        if like is not None:
            self.shape = like.shape
            self.tensor_shape = like.tensor_shape

        self.dtype = np.dtype('float64')

    def __copy__(self) -> 'KroneckerIdentity':
        new = KroneckerIdentity(like=self)
        new.factor = self.factor
        return new

    def __deepcopy__(self, memodict=None) -> 'KroneckerIdentity':
        new = KroneckerIdentity(like=self)
        new.factor = self.factor
        return new

    def __pow__(self, power: numeric, modulo=None) -> 'KroneckerIdentity':

        if power < 0:
            raise NotImplementedError

        new = KroneckerIdentity(like=self)
        new.factor = self.factor ** power
        return new

    def __matmul__(self, other: Union[KroneckerOperator, ndarray]) -> Union[KroneckerOperator, ndarray]:

        # if other is another KroneckerOperator, modify the factor and return
        if isinstance(other, KroneckerOperator):
            self.check_operators_consistent(self, other)
            new = other.copy()
            new.factor *= self.factor
            return new

        else:
            return super().__matmul__(other)

    def __mul__(self, other: Union['KroneckerOperator', numeric]) -> KroneckerOperator:

        if isinstance(other, KroneckerOperator):

            self.check_operators_consistent(self, other)

            if isinstance(other, (KroneckerProduct, KroneckerSum, KroneckerDiag, OperatorSum, OperatorProduct)):
                return self.factor * KroneckerDiag(ten(other.diag(), shape=self.tensor_shape))

            elif isinstance(other, KroneckerIdentity):
                return self.factor * other.factor * KroneckerIdentity(like=self)

            else:
                raise TypeError('A KroneckerDiag cannot be multiplied element-wise onto this operator')

        # otherwise other should be a scalar, handled in the base class
        else:
            return super().__mul__(other)

    def operate(self, other: ndarray) -> ndarray:
        return self.factor * multiply_tensor_identity(self.shape, other)

    def inv(self) -> 'KroneckerIdentity':
        return self.factor ** -1 * KroneckerIdentity(like=self)

    @property
    def T(self) -> 'KroneckerIdentity':
        return self

    def conj(self) -> 'KroneckerIdentity':
        return np.conj(self.factor) * self

    def diag(self) -> ndarray:
        return self.factor * np.ones(len(self))

    def to_array(self) -> ndarray:
        return self.factor * np.eye(self.shape[0])

    def __repr__(self) -> str:
        return 'KroneckerIdentity({})'.format(self.shape)

    def __str__(self) -> str:
        return 'KroneckerIdentity({})'.format(self.shape)
