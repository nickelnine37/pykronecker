from __future__ import annotations

import numpy as np
from numpy import ndarray
from typing import Union, List
from numpy.linalg import inv

# import kronecker as kron
from pykronecker.base import KroneckerOperator
from pykronecker.types import numeric
from pykronecker.utils import multiply_tensor_product, multiply_tensor_sum, multiply_tensor_diag, multiply_tensor_identity, kronecker_product_literal, kronecker_sum_literal, kronecker_diag_literal


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
        return self.factor * multiply_tensor_diag(self.A, other)

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


class KroneckerIdentity(KroneckerOperator):

    def __init__(self, size: int | tuple=None, like: KroneckerOperator=None):
        """
        Initiaise an Identity matrix using size parameter, or alternatively pass another operator of the same size.
        """

        if size is None and like is None:
            raise ValueError('Either shape or like must be passed')

        if size is not None:

            if isinstance(size, (tuple, list, ndarray)):
                assert len(size) == 2, 'parameter `size` should be length 2'
                assert size[0] == size[1], 'operator shape should be square'
                self.shape = tuple(size)

            elif isinstance(size, (int, np.int32, np.int64)):
                self.shape = (size, size)

        if like is not None:
            self.shape = like.shape

    def __copy__(self) -> 'KroneckerIdentity':
        new = KroneckerIdentity(like=self)
        new.factor = self.factor
        return new

    def __deepcopy__(self, memodict={}) -> 'KroneckerIdentity':
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

    def __mul__(self, other: Union['KroneckerDiag', numeric]) -> KroneckerOperator:

        if isinstance(other, KroneckerDiag):
            self.check_operators_consistent(self, other)
            return self.factor * other.factor * KroneckerDiag(other.A)

        elif isinstance(other, KroneckerIdentity):
            self.check_operators_consistent(self, other)
            return self.factor * other.factor * KroneckerIdentity(like=self)

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

    def to_array(self) -> ndarray:
        return self.factor * np.eye(self.shape[0])

    def __repr__(self) -> str:
        return 'KroneckerIdentity({})'.format(self.shape)

    def __str__(self) -> str:
        return 'KroneckerIdentity({})'.format(self.shape)


