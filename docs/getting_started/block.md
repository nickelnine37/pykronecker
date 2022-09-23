# Block Operators

Operators can be composed together to form new higher-order block matrices. For example, consider the following block matrix. 
$$
\mathbf{M} = 
\begin{bmatrix}
\mathbf{A}^{(1)} \otimes \mathbf{A}^{(2)} & \mathbf{I} \\
\mathbf{I} & \mathbf{B}^{(1)} \oplus \mathbf{B}^{(2)}
\end{bmatrix}
$$
We can create a representation of this matrix using the [`KroneckerBlock`](../../api/kroneckerblock) class. 

```python
import numpy as np
from pykronecker import KroneckerProduct, KroneckerSum, KroneckerIdentity

Ns = (200, 300)

As = [np.random.normal(size=(N, N)) for N in Ns]
Bs = [np.random.normal(size=(N, N)) for N in Ns]

M11 = KroneckerProduct(As)
M12 = KroneckerIdentity(tensor_shape=Ns)
M21 = KroneckerIdentity(tensor_shape=Ns)
M22 = KroneckerSum(Bs)

M = KroneckerBlock([[M11, M12], 
                    [M21, M22]])
```

This operator `M` can now act on vectors of shape `(200 * 300 * 2, )`

```python
x = np.random.normal(size=200 * 300 * 2)

print(M @ x)
```

Block operators can also be composed of a mixture of `KroneckerOperators` and NumPy arrays. Consider a the following block matrix
$$
\mathbf{L} = 
\begin{bmatrix}
\mathbf{A}^{(1)} \otimes \mathbf{A}^{(2)} & \mathbf{C} \\
\mathbf{C}^\top & \mathbf{I}_{(5 \times 5)}
\end{bmatrix}
$$
This operator, which acts on vectors of shape `(200 * 300 + 5)`, can be constructed as follows. 

```python
M11 = KroneckerProduct(As)
C = np.random.normal(size=(200 * 300, 5))
I = np.eye(5)

L = KroneckerBlock([[M11, C], 
                    [C.T, I]])

y = np.random.normal(size=(200 * 300 * 5))

print(L @ y)
```

For convenience, there is also a `KroneckerBlockDiag` class which creates block diagonal operators. For example, consider the following block diagonal matrix. 
$$
\mathbf{P} = 
\begin{bmatrix}
\mathbf{A}^{(1)} \otimes \mathbf{A}^{(2)} & \mathbf{0} \\
\mathbf{0}^\top & \mathbf{I}_{(5 \times 5)}
\end{bmatrix}
$$
This can be created as follows. 

```python
M11 = KroneckerProduct(As)
I = np.eye(5)

L = KroneckerBlockDiag([M11, I])

y = np.random.normal(size=(200 * 300 * 5))

print(L @ y)
```

