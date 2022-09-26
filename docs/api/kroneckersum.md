# KroneckerSum

Used to represent the object \(A^{(1)} \oplus A^{(2)} \oplus ... \oplus A^{(n)}\), that is the Kronecker product of \(N\) square matrices.

```python
def __init__(self, As: List[ndarray])
```

Initialise by passing in a sequence of square Numpy or Jax arrays

## Attributes

| Attribute      | Type         | Description                                                  |
| -------------- | ------------ | ------------------------------------------------------------ |
| `factor`       | `float, int` | A numerical factor multiplying the whole matrix              |
| `shape`        | `tuple[int]` | The total shape of the implicit matrix                       |
| `tensor_shape` | `tuple[int]` | The shape of tensors which this operator is expected to act on |
| `dtype`        | `np.dtype`   | The data type of the implicit matrix                         |



## Methods

see [`KroneckerOperator`](../kroneckeroperator)

### Example

```python
from pykronecker import KroneckerSum
import numpy as np

Ns = (20, 30, 40)
As = [np.random.normal(size=N) for N in Ns]
X = np.random.normal(size=Ns)

M = KroneckerSum(As)
print(M @ X)
```