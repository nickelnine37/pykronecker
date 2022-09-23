# Deriving New Operators

Many basic algebraic operations can be performed on a pair of [`KroneckerOperator`](../../api/kroneckeroperator)s. Doing so will create a new composite operator. 

For example, two or more operators can be added or subtracted from one another. 

```python
import numpy as np
from pykronecker import KroneckerProduct, KroneckerSum, KroneckerDiag

Ns = (30, 40, 50)

KP = KroneckerProduct([np.random.normal(size=(Ni, Ni)) for Ni in Ns])
KS = KroneckerSum([np.random.normal(size=(Ni, Ni)) for Ni in Ns])

X = np.random.normal(size=Ns)

A = KP + KS
print(A @ X)
```

Operators, just like regular matrices, can also also be multiplied together using the `@` syntax. 

```python
B = KP @ KS

print(B @ X)
```

Simple operators can also be multiplied element-wise using `*`. 

```python
C = KP * KS

print(C @ X)
```

All operators can be multiplied or divided by a number,

```
D = 5 * KP

print(D @ X)
```

and all operators can be transposed with `.T` or conjugate-transposed with `.H`.

```python
E = KS.T

print(E @ X)
```

Any of these operations can be combined in a single line to derive complex composite operators.

```python
F = 2 * KP.T @ KS + (KP / 2 - KS.T) @ KP

print(F @ X)
```

