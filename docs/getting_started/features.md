# Other Features

A number of other features are available for [`KroneckerOperator`](../../api/kroneckeroperator) instances. 

Operators can be summed down axis 0, or 1 or over the whole matrix. 

```python
import numpy as np
from pykronecker import KroneckerProduct

Ns = (5, 6, 7)
As = [np.random.normal(size=(Ni, Ni)) for Ni in Ns]

KP = KroneckerProduct(As)

print(KP.sum(0))
print(KP.sum(1))
print(KP.sum())
```

Operators can also be turned into a literal array using `.to_array()`. However, this should only be used for testing purposed on small matrices. Attempting to do this on a large operator will likely give a memory error. 

```python
print(KP.to_array())
```

[`KroneckerProduct`](../../api/kroneckerproduct), [`KroneckerDiag`](../../api/kroneckerdiag) and [`KroneckerIdentity`](../../api/kroneckeridentity) operators, as well as products and Block diagonal matrices constructed with these operators, can be inverted using `.inv()`. 

```python
KPi = KP.inv()

print(KPi @ np.random.normal(size=Ns))
```

The diagonal of an operator can be computed by running `.diag()`. 

```python
KP_diag = KP.diag()
```

A copy of an operator can be created with `.copy()`. 

```python
KP_copy = KP.copy()
```

However, this object still points to the same underlying arrays as `KP`. To completely copy the object, along with the underlying arrays, use `.deepcopy()`. 

```python
KP_deep = KP.deepcopy()
```


