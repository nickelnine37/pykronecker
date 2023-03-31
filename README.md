![Logo](https://raw.githubusercontent.com/nickelnine37/pykronecker/main/assets/logo.png)

[![DOI](https://joss.theoj.org/papers/10.21105/joss.04900/status.svg)](https://doi.org/10.21105/joss.04900)
![Tests](https://github.com/nickelnine37/pykronecker/actions/workflows/tests.yml/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/nickelnine37/pykronecker/badge.svg)](https://coveralls.io/github/nickelnine37/pykronecker)
[![Documentation Status](https://readthedocs.org/projects/pykronecker/badge/?version=latest)](https://pykronecker.readthedocs.io/en/latest/?badge=latest)

Check out the full documentation and install instructions [here](https://pykronecker.readthedocs.io/en/latest/) :)

# Overview

PyKronecker is a library for manipulating matrices which have a [Kronecker product](https://en.wikipedia.org/wiki/Kronecker_product) structure. Systems involving Kronecker products arise in many areas of applied mathematics and statistics. The aim of this library is to provide a clean interface for dealing with such systems, combining lazy evaluation and algebraic tricks to deliver large savings in terms of both memory and execution time. 

# Installation

Installation on Windows, OSX and Linux can be performed by running

```
pip3 install pykronecker
```

This will install the vanilla version of the library, with support for NumPy arrays only. Linux users have the additional option of installing PyKronecker with [Jax](https://jax.readthedocs.io/en/latest/index.html) support. The benefit of this is significantly faster runtimes, even when working with NumPy arrays only, due to Jax's JIT complier. This can be installed by running

```
pip3 install "pykronecker[jax]"
```

For Linux users with an Nvidia graphics card, PyKronecker is also compatible with the GPU and TPU version of Jax. However, since this relies on CUDA and cuDNN, it is recommended to follow the instructions [here](https://github.com/google/jax#installation) to install Jax first. 

# Usage

The concept of this library is to create instances of a `KroneckerOperator` class, which can be broadly treated as if it is a square numpy array. These objects are designed to be used with the `@` syntax for matrix multiplication. 

## Basic operators

### KroneckerProduct

Create a `KroneckerProduct` from two or more square NumPy/Jax arrays. These can be real or complex valued. 

```python
import numpy as np
from pykronecker import KroneckerProduct

A = np.random.normal(size=(5, 5))
B = np.random.normal(size=(6, 6))

KP = KroneckerProduct([A, B])
```

This object can operate on both vectors of shape `(5 * 6, )` and tensors of shape `(5, 6)` using the `@` syntax for matrix multiplication. The returned array will be of the same shape.

```python
x = np.random.normal(size=5 * 6)
X = x.reshape(5, 6)

assert np.allclose(KP @ x, (KP @ X).ravel())
```

### KroneckerSum

A `KronekerSum` can be created and used in much the same way.
```python
import numpy as np
from pykronecker import KroneckerSum

A = np.random.normal(size=(5, 5))
B = np.random.normal(size=(6, 6))
x = np.random.normal(size=5 * 6)

KS = KroneckerSum([A, B])
print(KS @ x)
```

### KroneckerDiag

`KroneckerDiag` provides support for diagonal matrices, and can be created by passing a tensor of the appropriate size. This creates, in effect, a matrix with the vectorized tensor along the diagonal. 

```python
import numpy as np
from pykronecker import KroneckerDiag

D = np.random.normal(size=(5, 6))
x = np.random.normal(size=5 * 6)

KD = KroneckerDiag(D)
print(KD @ x)
```

### KroneckerIdentity

Finally, `KroneckerIdentity` creates the identity matrix, which can be instantiated by passing another operator of the same size, or the shape of tensors the operator is expected to act on. 

```python
import numpy as np
from pykronecker import KroneckerIdentity, KroneckerDiag

# create another KroneckerDiag operator
D = np.random.normal(size=(5, 6))
KD = KroneckerDiag(D)

# create a KroneckerIdentity by passing `like` parameter
KI1 = KroneckerIdentity(like=KD)

# create KroneckerIdentity by passing `tensor_shape` parameter
KI2 = KroneckerIdentity(tensor_shape=(5, 6))

x = np.random.normal(size=5 * 6)

assert np.allclose(KI1 @ x, x)
assert np.allclose(KI2 @ x, x)
```

## Deriving new operators

All four of these objects can be added or multiplied together arbitrarily to create new composite operators. In this way, they can be treated similarly to literal NumPy arrays. 

```python
import numpy as np
from pykronecker import *

A = np.random.normal(size=(5, 5))
B = np.random.normal(size=(6, 6))
D = np.random.normal(size=(5, 6))
x = np.random.normal(size=5 * 6)

KP = KroneckerProduct([A, B])
KS = KroneckerSum([A, B])
KD = KroneckerDiag(D)
KI = KroneckerIdentity(like=KP)

# create a new composite operator!
new_operator1 = KP @ KD + KS - KI

print(new_operator1 @ x)
```

Other possible operations include transposing with `.T`, and multiplying/dividing by a scalar. 

```python
new_operator2 = 5 * KP.T - KS / 2

print(new_operator2 @ x)
```

Many basic operators can also be multipled element-wise just as with NumPy arrays. 

```python
new_operator3 = KS * KP

print(new_operator3 @ x)
```

Some operators (notably, not `KroneckerSum`s) can be raised to a power element-wise

```python
new_operator4 = KP ** 2

print(new_operator4 @ x)
```


## Block operators

Block operators are composed of smaller operators which have been stacked into a set of blocks. In the example below, we create a new block operator `KB` which is composed of four other block operators. 

```python
import numpy as np
from pykronecker import *

A = np.random.normal(size=(5, 5))
B = np.random.normal(size=(6, 6))
D = np.random.normal(size=(5, 6))

KP = KroneckerProduct([A, B])
KS = KroneckerSum([A, B])
KD = KroneckerDiag(D)
KI = KroneckerIdentity(like=KP)

# Create a block of pure KroneckerOperators
KB1 = KroneckerBlock([[KP, KD], 
                      [KI, KS]])

x1 = np.random.normal(size=5 * 6 * 2)
print(KB1 @ x1)
```

We can also create block operators that contain a mixture of `KroneckerOperator`s and NumPy arrays

```python
# Create a block with a mixture of KroneckerOperators and ndarrays

M11 = KP
M12 = np.ones((5 * 6, 5))
M21 = np.random.normal(size=(5, 5 * 6))
M22 = np.eye(5)

KB2 = KroneckerBlock([[M11, M12], 
                      [M21, M22]])

x2 = np.random.normal(size=5 * 6 + 5)
print(KB2 @ x2)

```
 
Block diagonal matrices can also be created in a similar way 

```python
from pykronecker import KroneckerBlockDiag

KBD = KroneckerBlockDiag([KP, KS])

x3 = np.random.normal(size=5 * 6 * 2)
print(KBD @ x3)
```


## Other features

For operators that are products of `KroneckerProduct`s, `KroneckerDiag`s, or `KroneckerIdentity`s, we can find the inverse with `.inv()`.

```python
import numpy as np
from pykronecker import *

A = np.random.normal(size=(5, 5))
B = np.random.normal(size=(6, 6))
D = np.random.normal(size=(5, 6))
x = np.random.normal(size=5 * 6)

KP = KroneckerProduct([A, B])
KS = KroneckerSum([A, B])
KD = KroneckerDiag(D)
KI = KroneckerIdentity(like=KP)

# find the inverse
M = (KP @ KD).inv()
print(M @ x)
```

Basic indexing is supported for all operators. This will return literal numpy arrays. [Advanced indexing](https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing) is not supported.

```python
J = KP.T + KS @ KD

print(J[0])
print(J[2:5])
print(J[2:8:3])
print(J[:, 2])
print(J[:, 2:5])
print(J[:, 2:8:3])
print(J[2, 5])
print(J[2:5, 2:8:3])
print(J[:])
print(J[:, :])
```

Summing down an axis or over the whole matrix is supported for any opertor.

```python
print(J.sum(0))
print(J.sum(1))
print(J.sum())
```

Any operator can also be converted to a literal array. This should only be used for small test purposes, as the arrays created can be very large. 

```python
print(J.to_array())
```

The matrix diagonal of most operators can be found with `.diag()`. This returns a one-dimensional array. 

```python
print(J.diag())
```

The conjugate transpose of any complex operator can be found with `.H`

```python

A_ = np.random.normal(size=(5, 5)) + 1j * np.random.normal(size=(5, 5))
B_ = np.random.normal(size=(6, 6)) + 1j * np.random.normal(size=(6, 6))

KP_ = KroneckerProduct([A_, B_])

print(KP_.H @ x)
```

## Use with JAX

Operators and tensors can also be created from Jax arrays for accelerated computation when the `pykronecker[jax]` extra has been installed. Note that this is only available on Linux and MacOS.  

```python
import numpy as np
import jax.numpy as jnp
from pykronecker import KroneckerProduct

A = jnp.asarray(np.random.normal(size=(5, 5)))
B = jnp.asarray(np.random.normal(size=(6, 6)))
x = jnp.asarray(np.random.normal(size=5 * 6))

KP = KroneckerProduct([A, B])

print(KP @ x)
```
