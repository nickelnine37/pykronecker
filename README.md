![Logo](https://raw.githubusercontent.com/nickelnine37/pykronecker/main/assets/logo.png)

![Tests](https://github.com/nickelnine37/pykronecker/actions/workflows/tests.yml/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/nickelnine37/pykronecker/badge.svg)](https://coveralls.io/github/nickelnine37/pykronecker)

# Overview

This library contains tools for performing efficient matrix operations with [Kronecker products](https://en.wikipedia.org/wiki/Kronecker_product). The Kronecker product between two square matrices *A* and *B* is constructed as follows. 

![Math](https://raw.githubusercontent.com/nickelnine37/pykronecker/main/assets/kronecker_product.png)

If *A* has size (*n* x *n*), and *B* has size (*m* x *m*), then their kronecker product has size (*mn* x *mn*). Furthermore, the result of taking the Kronecker product of three matrices *A*, *B* and *C*, where *C* has shape (*p* x *p*), is a matrix of shape (*nmp* x *nmp*) and so on. The aim of this package is to efficiently caluclate matrix-vector multiplications in this expanded space without ever needing to create or operate with these potentially vast matrices directly. 


# Installation

```
pip3 install pykronecker
```

# Usage

We create instances of a `KroneckerOperator` class, which can be broadly treated as if it is a square numpy array. These objects are designed to be used with the `@` syntax for matrix multiplication. 

## Basic operators

Create a `KroneckerProduct` from two or more square numpy arrays. These can be real or complex valued. 

```python
import numpy as np
from pykronecker import KroneckerProduct

A = np.random.normal(size=(5, 5))
B = np.random.normal(size=(6, 6))
C = KroneckerProduct([A, B])
```

This object can operate on both vectors of shape `(5 * 6, )` and tensors of shape `(5, 6)` using the `@` syntax for matrix multiplication. The returned array will be of the same shape.  

```python
x = np.random.normal(size=5 * 6)
X = np.random.normal(size=(5, 6))
print(C @ x)
print(C @ X)
```

A `KronekerSum` can be created and used in much the same way. The Kronecker sum of two square matrices *A* and *B* is defined as folows.

![Math](https://raw.githubusercontent.com/nickelnine37/pykronecker/main/assets/kronecker_sum.png)

```python
from pykronecker import KroneckerSum

D = KroneckerSum([A, B])
print(D @ x)
```

`KroneckerDiag` provides support for diagonal matrices.

```python
from pykronecker import KroneckerDiag

E = KroneckerDiag(np.random.normal(size=5 * 6))
print(E @ x)
```

Finally, `KroneckerIdentity` creates the identity matrix, which can be instantiated by passing another operator of the same size, or the shape of tensors it should act on. 

```python
from pykronecker import KroneckerIdentity

I1 = KroneckerIdentity(like=E)
I2 = KroneckerIdentity(tensor_shape=(5, 6))
print(I1 @ x, I2 @ x)
```

## Deriving new operators

All four of these objects can be added or multiplied together arbitrarily to create new composite operators. In this way, they can be treated similarly to literal NumPy arrays. 

```python
F = C @ D + C @ E
print(F @ x)
```

Other possible operations include transposing with `.T`, and multiplying/dividing by a scalar. 

```python
G = 2 * F.T + E / 5 
print(G @ x)
```

Many basic operators can also be multipled element-wise just as with NumPy arrays. 

```python
H = C * D
print(H @ x)
```

## Block operators

We can create block operators by stacking together any mixture of `KroneckerOperator`s and/or NumPy arrays. 

```python
from pykronecker import KroneckerBlock

# Block of pure KroneckerOperators
M = KroneckerBlock([[C, D], 
                    [E, F]])

print(M @ np.random.normal(size=5 * 6 * 2))

# Block with mixture of KroneckerOperators and ndarrays
N11 = E
N12 = np.ones((5 * 6, 5))
N21 = np.random.normal(size=(5, 5 * 6))
N22 = np.eye(5)

N = KroneckerBlock([[N11, N12], 
                    [N21, N22]])

print(N @ np.random.normal(size=5 * 6 * 2))
```

Block diagonal matrices can also be created in a similar way 

```python
from pykronecker import KroneckerBlockDiag

J = KroneckerBlockDiag([E, F])
print(M @ np.random.normal(size=5 * 6 * 2))
```


## Other features

For operators that are products of `KroneckerProduct`s, `KroneckerDiag`s, or `KroneckerIdentity`s, we can find the inverse with `.inv()`.

```python
H = (C @ E).inv()
print(H @ x)
```

Summing down an axis or over the whole matrix is supported.

```python
print(F.sum(0))
print(F.sum(1))
print(F.sum())
```

As is conversion to a literal array 

```python
print(H.to_array())
```

Element-wise power operation can be used with `**`

```python
print(C ** 2)
```

The matrix diagonal can be found with `.diag()`

```python
print(C.diag())
```

Th conjugate transpose with `.H`

```python
print(C.H)
```