# Other Features

A number of other features are available for [`KroneckerOperator`](../../api/kroneckeroperator) instances. To demonstrate these, lets begin by creating some operators

```python
import numpy as np
from pykronecker import KroneckerProduct, KroneckerSum

Ns = (5, 6, 7)
As = [np.random.normal(size=(Ni, Ni)) for Ni in Ns]

KP = KroneckerProduct(As)
KS = KroneckerSum(As)
M = KP + KS
```

## Literal operations

There are a number of operations for finding literal representations of the whole operator or parts of the operator. 


The diagonal of any operator can be computed by running `.diag()`. 

```python
print(M.diag())
```

We can also use indexing to extract parts of an operator. This works just like indexing a two-dimensinoal numpy array.

```python
print(M[0])
print(M[2:5])
print(M[2:8:3])
print(M[:, 2])
print(M[:, 2:5])
print(M[:, 2:8:3])
print(M[2, 5])
print(M[2:5, 2:8:3])
print(M[:])
print(M[:, :])
```

However, [advanced indexing](https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing) routines such as boolean indexing are not supported. 

In addition to using indexing, the full operator can also be turned into a literal array using `.to_array()`. However, this should only be used for testing purposed on small matrices. Attempting to do this on a large operator will likely give a memory error. 

```python
print(M.to_array())
```

## Summation

Operators can be summed down axis 0, or 1 or over the whole matrix. 

```python
print(M.sum(0))
print(M.sum(1))
print(M.sum())
```

## Inverting an operator

[`KroneckerProduct`](../../api/kroneckerproduct), [`KroneckerDiag`](../../api/kroneckerdiag) and [`KroneckerIdentity`](../../api/kroneckeridentity) operators, as well as products and Block diagonal matrices constructed with these operators, can be inverted using `.inv()`. 

```python
KPi = KP.inv()

print(KPi @ np.random.normal(size=Ns))
```


## Copying operators

A copy of an operator can be created with `.copy()`. 

```python
KP_copy = KP.copy()
```

However, this object still points to the same underlying arrays as `KP`. To completely copy the object, along with the underlying arrays, use `.deepcopy()`. 

```python
KP_deep = KP.deepcopy()
```


