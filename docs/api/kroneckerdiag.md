# KroneckerDiag

Used to represent a general diagonal matrix of size \(N^{(1)} \times N^{(2)} \times ... \times N^{(n)}\)

### __init__

```python
def __init__(self, A: ndarray))
```

Initialise with a tensor `A` of shape \(N^{(1)} \times N^{(2)} \times ... \times N^{(n)}\), which is a NumPy or Jax array


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
from pykronecker import KroneckerDiag
import numpy as np

Ns = (20, 30, 40)
A = np.random.normal(size=Ns)
X = np.random.normal(size=Ns)

M = KroneckerDiag(A)
print(M @ X)
```