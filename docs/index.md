![Logo](https://raw.githubusercontent.com/nickelnine37/pykronecker/main/assets/logo.png)

![Tests](https://github.com/nickelnine37/pykronecker/actions/workflows/tests.yml/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/nickelnine37/pykronecker/badge.svg)](https://coveralls.io/github/nickelnine37/pykronecker)

[GitHub](https://github.com/nickelnine37/pykronecker) \(\cdot\) [PyPi](https://pypi.org/project/pykronecker/) \(\cdot\) [libraries.io](https://libraries.io/pypi/pykronecker)

PyKronecker is a library for manipulating matrices which have a [Kronecker product](https://en.wikipedia.org/wiki/Kronecker_product) structure.  Systems involving Kronecker products arise in many areas of applied mathematics and statistics. The aim of this library is to provide a clean interface for dealing with such systems, combining lazy evaluation and algebraic tricks to deliver large savings in terms of both memory and execution time. 

## Quickstart

Create a [`KroneckerProduct`](https://pykronecker.readthedocs.io/en/latest/api/kroneckerproduct/) operator from two or more square arrays. These may be real or complex-valued. 

```python
import numpy as np
from pykronecker import KroneckerProduct

Ns = [30, 40, 50]
As = [np.random.normal(size=(Ni, Ni)) for Ni in Ns]

A = KroneckerProduct(As)
```

This object can be multiplied onto a vector of shape `(30 * 40 * 50, )` or a tensor of shape `(30, 40, 50)` using the `@` syntax for matrix multiplication. The returned array will be the same size as the input array. 

```python
X = np.random.normal(size=(30 * 40 * 50))
x = np.random.normal(size=(30, 40, 50))

print(A @ X, A @ x)
```

A [`KroneckerSum`](https://pykronecker.readthedocs.io/en/latest/api/kroneckersum/) operator can be created and used in much the same way. 

```python
from pykronecker import KroneckerSum

B = KroneckerSum(As)

print(A @ X, A @ x)
```

These objects can be added, scaled, and transposed arbitrarily to create new composite operators.  

```
C = 2 * A - 0.5 * B.T

print(C @ X, C @ x)
```



