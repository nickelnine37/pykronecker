# KroneckerBlock

Create a general block operator. Items in the block can be arrays or operators.

### __init__

```python
def __init__(self, blocks: List[list]):
```

e.g.

```python
blocks = [[A11, A12, A13]
          [A21, A22, A23]
          [A31, A32, A33]]
```

where each item is a [`KroneckerOperator`](../kroneckeroperator), NumPy array, or Jax array of the appropriate shape. 

## Attributes

| Attribute      | Type         | Description                                                  |
| -------------- | ------------ | ------------------------------------------------------------ |
| `factor`       | `float, int` | A numerical factor multiplying the whole matrix              |
| `shape`        | `tuple[int]` | The total shape of the implicit matrix                       |
| `tensor_shape` | `tuple[int]` | The shape of tensors which this operator is expected to act on |
| `dtype`        | `np.dtype`   | The data type of the implicit matrix                         |

## Methods

see [`KroneckerOperator`](../kroneckeroperator)

## Example

```python
import numpy as np
from pykronecker import KroneckerProduct, KroneckerSum, KroneckerIdentity, KroneckerBlock

Ns = (20, 30, 40)

As = [np.random.normal(size=(N, N)) for N in Ns]
Bs = [np.random.normal(size=(N, N)) for N in Ns]

M11 = KroneckerProduct(As)
M12 = KroneckerIdentity(tensor_shape=Ns)
M21 = KroneckerIdentity(tensor_shape=Ns)
M22 = KroneckerSum(Bs)

M = KroneckerBlock([[M11, M12],
                    [M21, M22]])

x = np.random.normal(size=20 * 30 * 40 * 2)

print(M @ x)
```

