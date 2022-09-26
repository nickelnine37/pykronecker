# KroneckerOperator

This is the abstract base class from which all other operators inherit. It should never be instantiated directly. 

## Attributes

| Attribute      | Type         | Description                                                  |
| -------------- | ------------ | ------------------------------------------------------------ |
| `factor`       | `float, int` | A numerical factor multiplying the whole matrix              |
| `shape`        | `tuple[int]` | The total shape of the implicit matrix                       |
| `tensor_shape` | `tuple[int]` | The shape of tensors which this operator is expected to act on |
| `dtype`        | `np.dtype`   | The data type of the implicit matrix                         |



## Methods

### __copy__

```python
@abstractmethod
def __copy__(self) -> KroneckerOperator:
```

Create a shallow copy of a Kronecker object. This does not copy any of the underlying arrays, but means,
for example, we can have kronecker objects with the same underlying arrays but different factors. This
needs to be implemented by subclasses.
### __deepcopy__

```python
@abstractmethod
def __deepcopy__(self, memodict=None) -> KroneckerOperator:
```

Create a deep copy of a Kronecker object. This copies the data in the underlying arrays to create
a totally independent object. This needs to be implemented by subclasses.
### __pow__

```python
@abstractmethod
def __pow__(self, power: numeric, modulo=None) -> KroneckerOperator:
```

Element-wise power operation. Only works for KroneckerProducts and KroneckerDiags.
### operate

```python
@abstractmethod
def operate(self, other: ndarray) -> ndarray:
```

This key method should describe how the Kronecker object acts on a Tensor/vector. This is where subclasses should
implement their efficient versions of matrix-vector multiplication.
### inv

```python
@abstractmethod
def inv(self) -> KroneckerOperator:
```

Inverse method. Use with caution.
### T

```python
@property
@abstractmethod
def T(self) -> KroneckerOperator:
```

Return a copy of the object transposed.
### to_array

```python
@abstractmethod
def to_array(self) -> ndarray:
```

Turn into a literal array. Use with caution!
### diag

```python
@abstractmethod
def diag(self) -> ndarray:
```

Return a vector representing the diagonal of the operator
### conj

```python
@abstractmethod
def conj(self) -> KroneckerOperator:
```

Return the operator with the complex conjugate applied element-wise
### __repr__

```python
@abstractmethod
def __repr__(self) -> str:
```

Subclasses should define their own repr
### __add__

```python
def __add__(self, other: KroneckerOperator) -> KroneckerOperator:
```

Overload the addition method. This is used to sum together KroneckerOperators and as such
`other` must be an instance of a KroneckerOperator, and not an array or other numeric type.
### __radd__

```python
def __radd__(self, other: KroneckerOperator) -> KroneckerOperator:
```

Order does not matter, so return __add__
### __sub__

```python
def __sub__(self, other: KroneckerOperator) -> KroneckerOperator:
```

Overload the subtraction method. Simply scale `other` by negative one and add
### __mul__

```python
def __mul__(self, other: Union[KroneckerOperator, numeric]) -> KroneckerOperator:
```

Multiply the linear operator element-wise. As with numpy arrays, the * operation defaults to element-wise
(Hadamard) multiplication, not matrix multiplication. For numbers, this is a simple scalar multiple. For
Kronecker objects, we can only define the behaviour efficiently for KroneckerProducts and KroneckerDiags,
which is implemented in the respective subclass.
### __rmul__

```python
def __rmul__(self, other: Union[KroneckerOperator, numeric]) -> KroneckerOperator:
```

Hadamard and scalar multiples are commutative
### __truediv__

```python
def __truediv__(self, other: numeric) -> KroneckerOperator:
```

Self-explanatory, but only works for numbers.
### H

```python
@property
def H(self):
```

Return the Hermitian conjugate (conjugate transpose) of the operator. For real operators,
this is equivalent to the transpose.
### __matmul__

```python
def __matmul__(self, other: Union[KroneckerOperator, ndarray]) -> Union[KroneckerOperator, ndarray]:
```

Overload the matrix multiplication method to use these objects with the @ operator.
### __rmatmul__

```python
def __rmatmul__(self, other: Union[KroneckerOperator, ndarray]) -> Union[KroneckerOperator, ndarray]:
```

Define reverse matrix multiplication in terms of transposes
### __pos__

```python
def __pos__(self) -> KroneckerOperator:
```

`+Obj == Obj`
### __neg__

```python
def __neg__(self) -> KroneckerOperator:
```

`-Obj == (-1) * Obj`
### __len__

```python
def __len__(self) -> int:
```

Get the length of the object.
### __str__

```python
def __str__(self) -> str:
```

Fallback string operator
### __array_ufunc__

```python
def __array_ufunc__(self, method, *inputs, **kwargs) -> Union[KroneckerOperator, ndarray]:
```

Override the numpy implementation of matmul, so that we can also use
this function rather than just the @ operator.

E.g.
`KroneckerProduct(A, B, C) @ vec(X) == np.matmul(KroneckerProduct(A, B, C), vec(X))`

Note that
`KroneckerProduct(A, B, C) @ X !=  np.matmul(KroneckerProduct(A, B, C), X)`

and that it does not work with `np.dot()`
### sum

```python
def sum(self, axis: int | None=None) -> ndarray | float:
```

Sum the operator along one axis as if it is a matrix. Or None for total sum.
### copy

```python
def copy(self) -> KroneckerOperator:
```

Alias for __copy__
### deepcopy

```python
def deepcopy(self) -> KroneckerOperator:
```

Alias for __deepcopy__
### check_operators_consistent

```python
@staticmethod
def check_operators_consistent(A: KroneckerOperator, B: KroneckerOperator) -> bool:
```

Check whether two KroneckerOperators are mutually compatible. I.e, check that they have the same shape.
### check_valid_matrices

```python
@staticmethod
def check_valid_matrices(As: List[ndarray]) -> bool:
```

Check whether As contains a list of square ndarrays suitable for use in
either a Kronecker product or a Kronecker sum
