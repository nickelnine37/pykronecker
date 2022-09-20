from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy import ndarray
from typing import Union, List

from pykronecker.utils import numeric


"""
The class in this file is a base class for all Kronecker operators. Kronecker operators represent large matrices in a
compact form, and perform all multiplications onto vectors lazily and efficiently. Composite operators can be created
by treating Kronecker operators as if they are NumPy matrices. All operators support:

    * addition
    * matrix multiplication
    * multiplication/division by a scalar
    * summing along axes 0, 1 or both
    * transposing

using +, @, *, .sum() and .T respectively. Some further behaviours for certain operator types are implemented in the subclasses. 

"""


class KroneckerOperator(ABC):
    """
    Base class defining the behaviour of Kronecker-type operators. It should not be instantiated directly.
    """

    __array_priority__ = 10             # increase priority of class, so it takes precedence when mixing matrix multiplications with ndarrays
    factor: numeric = 1.0               # a scalar factor multiplying the whole operator
    shape: tuple[int, int] = (0, 0)     # full (N, N) operator shape
    tensor_shape: tuple = None          # the expected shape of tensors this operator acts on
    dtype = None                        # the dtype of the underlying matrices (or np.result_type if they differ)

    # ------------- ABSTRACT METHODS --------------
    # These should all be defined by subclasses

    @abstractmethod
    def __copy__(self) -> 'KroneckerOperator':
        """
        Create a shallow copy of a Kronecker object. This does not copy any of the underlying arrays, but means,
        for example, we can have kronecker objects with the same underlying arrays but different factors. This
        needs to be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def __deepcopy__(self, memodict=None) -> 'KroneckerOperator':
        """
        Create a deep copy of a Kronecker object. This copies the data in the underlying arrays to create
        a totally independent object. This needs to be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def __pow__(self, power: numeric, modulo=None) -> 'KroneckerOperator':
        """
        Element-wise power operation. Only works for KroneckerProducts and KroneckerDiags.
        """
        raise NotImplementedError

    @abstractmethod
    def operate(self, other: ndarray) -> ndarray:
        """
        This key method should describe how the Kronecker object acts on a Tensor/vector. This is where subclasses should
        implement their efficient versions of matrix-vector multiplication.
        """
        raise NotImplementedError

    @abstractmethod
    def inv(self) -> 'KroneckerOperator':
        """
        Inverse method. Use with caution.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def T(self) -> 'KroneckerOperator':
        """
        Return a copy of the object transposed.
        """
        raise NotImplementedError

    @abstractmethod
    def to_array(self) -> ndarray:
        """
        Turn into a literal array. Use with caution!
        """

        raise NotImplementedError

    @abstractmethod
    def diag(self) -> ndarray:
        """
        Return a vector representing the diagonal of the operator
        """
        raise NotImplementedError

    @abstractmethod
    def conj(self) -> 'KroneckerOperator':
        """
        Return the operator with the complex conjugate applied element-wise
        """
        raise NotImplementedError

    @abstractmethod
    def __repr__(self) -> str:
        """
        Subclasses should define their own repr
        """
        raise NotImplementedError

    # ----------- CONCRETE METHODS ----------
    # These define shared and default behaviours

    def __add__(self, other: 'KroneckerOperator') -> 'KroneckerOperator':
        """
        Overload the addition method. This is used to sum together KroneckerOperators and as such
        `other` must be an instance of a KroneckerOperator, and not an array or other numeric type.
        """

        from pykronecker.composite import OperatorSum

        if not isinstance(other, KroneckerOperator):
            raise TypeError('Kronecker operators can only be added to other Kronecker operators')

        return OperatorSum(self, other)

    def __radd__(self, other: 'KroneckerOperator') -> 'KroneckerOperator':
        """
        Order does not matter, so return __add__
        """
        return self.__add__(other)

    def __sub__(self, other: 'KroneckerOperator') -> 'KroneckerOperator':
        """
        Overload the subtraction method. Simply scale `other` by negative one and add
        """
        return self.__add__((-1.0) * other)

    def __mul__(self, other: Union['KroneckerOperator', numeric]) -> 'KroneckerOperator':
        """
        Multiply the linear operator element-wise. As with numpy arrays, the * operation defaults to element-wise
        (Hadamard) multiplication, not matrix multiplication. For numbers, this is a simple scalar multiple. For
        Kronecker objects, we can only define the behaviour efficiently for KroneckerProducts and KroneckerDiags,
        which is implemented in the respective subclass.
        """

        if isinstance(other, (int, float, complex, np.number)):
            # create a copy of the object rather than mutating the factor directly, which is cleaner and leads to less unexpected behaviour
            new = self.copy()
            new.factor = self.factor * other
            return new

        elif isinstance(other, KroneckerOperator):
            raise TypeError('Only KroneckerProducts and KroneckerDiags can be multiplied together element-wise')

        else:
            raise TypeError('General Kronecker operators can only be scaled by a number')

    def __rmul__(self, other: Union['KroneckerOperator', numeric]) -> 'KroneckerOperator':
        """
        Hadamard and scalar multiples are commutative
        """
        return self.__mul__(other)

    def __truediv__(self, other: numeric) -> 'KroneckerOperator':
        """
        Self-explanatory, but only works for numbers.
        """
        return self.__mul__(1.0 / other)

    @property
    def H(self):
        """
        Return the Hermitian conjugate (conjugate transpose) of the operator. For real operators,
        this is equivalent to the transpose.
        """
        return self.conj().T

    def __matmul__(self, other: Union['KroneckerOperator', ndarray]) -> Union['KroneckerOperator', ndarray]:
        """
        Overload the matrix multiplication method to use these objects with the @ operator.
        """

        from pykronecker.composite import OperatorProduct
        from pykronecker.operators import KroneckerIdentity

        if isinstance(other, ndarray):
            return self.operate(other)

        elif isinstance(other, KroneckerIdentity):
            new = self.copy()
            new.factor *= other.factor
            return new

        elif isinstance(other, KroneckerOperator):
            return OperatorProduct(self, other)

        else:
            raise TypeError(f'Objects in the matrix product must be Kronecker Operators or ndarrays, but this is a {type(other)}')

    def __rmatmul__(self, other: Union['KroneckerOperator', ndarray]) -> Union['KroneckerOperator', ndarray]:
        """
        Define reverse matrix multiplication in terms of transposes
        """

        if isinstance(other, ndarray):

            # we have a left-sided tensor multiplication
            if int(np.prod(other.shape)) == self.shape[0]:
                return self.T @ np.squeeze(other)

            # we have a left sided data matrix multiplication
            else:
                return (self.T @ other.T).T

        # we should never get KroneckerOperators here, as it will be handled by __matmul__
        else:
            raise TypeError('Objects in the matrix product must be Kronecker Operators or ndarrays')

    def __pos__(self) -> 'KroneckerOperator':
        """
        +Obj == Obj
        """
        return self

    def __neg__(self) -> 'KroneckerOperator':
        """
        -Obj == (-1) * Obj
        """
        return (-1) * self

    def __len__(self) -> int:
        return self.shape[0]

    def __str__(self) -> str:
        """
        Fallback string operator
        """
        return f'KroneckerOperator{self.shape}'

    def __array_ufunc__(self, method, *inputs, **kwargs) -> Union['KroneckerOperator', ndarray]:
        """
        Override the numpy implementation of matmul, so that we can also use
        this function rather than just the @ operator.

        E.g.
            KroneckerProduct(A, B, C) @ vec(X) == np.matmul(KroneckerProduct(A, B, C), vec(X))

        Note that
            KroneckerProduct(A, B, C) @ X !=  np.matmul(KroneckerProduct(A, B, C), X)

        and that it does not work with np.dot()
        """

        A, B = inputs[1], inputs[2]

        if method is np.matmul:

            if A is self:
                return self.__matmul__(B)

            if B is self:
                return self.__rmatmul__(A)

        elif method is np.multiply:

            if A is self:
                return self.__mul__(B)

            if B is self:
                return self.__mul__(A)

        else:
            raise NotImplementedError

    def quadratic_form(self, X: ndarray) -> float:
        """
        Compute the quadratic form vec(X).T @ self @ vec(X)
        """

        if not isinstance(X, ndarray):
            raise TypeError

        return (X * (self @ X)).sum()

    def sum(self, axis: int | None=None) -> ndarray | float:
        """
        Sum the operator along one axis as if it is a matrix. Or None for total sum.
        """

        ones = np.ones(self.shape[0])

        if axis is None:
            return self.quadratic_form(ones)

        elif axis == 1 or axis == -1:
            return self @ ones

        elif axis == 0:
            return self.T @ ones

        else:
            raise ValueError('Axis should be -1, 0, 1 or None')

    def copy(self) -> 'KroneckerOperator':
        return self.__copy__()

    def deepcopy(self) -> 'KroneckerOperator':
        return self.__deepcopy__()

    # ----------- STATIC METHODS -----------
    # group functions useful for subclasses

    @staticmethod
    def check_operators_consistent(A: 'KroneckerOperator', B: 'KroneckerOperator') -> bool:
        """
        Check whether two KroneckerOperators are mutually compatible. I.e, check that they have the same shape.
        """

        assert all(isinstance(C, KroneckerOperator) for C in [A, B]), f'All operators in this chain must be consistent, but they have types {type(A)} and {type(B)} respectively'
        assert A.shape == B.shape, f'All operators in this chain should have the same shape, but they have shapes {A.shape} and {B.shape} respectively'
        assert A.tensor_shape == B.tensor_shape, f'All operators in this chain should act on tensors of the same shape, but they act on {A.tensor_shape} and {B.tensor_shape} respectively'

        return True

    @staticmethod
    def check_valid_matrices(As: List[ndarray]) -> bool:
        """
        Check whether As contains a list of square ndarrays suitable for use in
        either a Kronecker product or a Kronecker sum
        """

        # assert all(isinstance(A, ndarray) for A in As)
        # code may be compatible with scipy sparse matrices...
        assert all(A.ndim == 2 for A in As)
        assert all(A.shape[0] == A.shape[1] for A in As)

        return True
