# KroneckerBlockDiag

Create a block diagonal operator. Items in the block can be arrays or operators.

### __init__

```python
def __init__(self, blocks: List[list]):
```

e.g.

```python
blocks = [A1, A2, A3] -> [[A1, 0, 0]
                          [0, A2, 0]
                          [0, 0, A3]]
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
from pykronecker import KroneckerBlockDiag, KroneckerProduct

Ns = (20, 30, 40)
As = [np.random.normal(size=(N, N)) for N in Ns]

M1 = KroneckerProduct(As)
I = np.eye(5)

L = KroneckerBlockDiag([M1, I])

y = np.random.normal(size=(200 * 300 * 5))

print(L @ y)
```
