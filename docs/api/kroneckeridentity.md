# KroneckerIdentity

Used to represent an identity matrix of size \(N^{(1)} \times N^{(2)} \times ... \times N^{(n)}\)

```python
def __init__(self, tensor_shape: tuple=None, like: KroneckerOperator=None)
```

Initialise by passing in either 

* `like`: another [`KroneckerOperator`](../kroneckeroperator) of the same size, or
* `tensor_shape`: the shape of tensors this operator should act on

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
from pykronecker import KroneckerIdentity
import numpy as np

Ns = (20, 30, 40)
I = KroneckerIdentity(tensor_shape=Ns)
X = np.random.normal(size=Ns)

print(I @ X)
```

