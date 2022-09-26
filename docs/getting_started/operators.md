# Basic Operators

PyKronecker provides four basic operators, which are all subclasses of the [`KroneckerOperator`](../api/kroneckeroperator) class. They are 

* [`KroneckerProduct`](#kroneckerproduct) 
* [`KroneckerSum`](#kroneckersum)
* [`KroneckerDiag`](#kroneckerdiag)
* [`KroneckerIdentity`](#kroneckeridentity)



## KroneckerProduct

The [`KroneckerProduct`](../../api/kroneckerproduct) class is used to represent the Kronecker product of a chain of square matrices \(\mathbf{A}^{(i)}\). 
$$
\mathbf{A} = \mathbf{A}^{(1)} \otimes \mathbf{A}^{(2)} \otimes \;\dots \; \otimes \mathbf{A}^{(n)} = \bigotimes_{i=1}^n \mathbf{A}^{(i)}
$$
It is instantiated by passing a sequence of square arrays containing real or complex-valued entries. 

```python
import numpy as np
from pykronecker import KroneckerProduct

Ns = (30, 40, 50)
As = [np.random.normal(size=(Ni, Ni)) for Ni in Ns]

KP = KroneckerProduct(As)
```

This object `KP` can now operate on either:

* vectors of shape `(30 * 40 * 50)`, or
* tensors of shape `(30, 40, 50)`

The resultant shape after `KP` operates on either of these objects will be of the same shape. 

```python
x = np.random.normal(size=(30 * 40 * 50))
X = np.random.normal(size=(30, 40, 50))

assert (KP @ x).shape == (30 * 40 * 50, )
assert (KP @ X).shape == (30, 40, 50)
```

Due to how this operation is handled under the hood, it is slightly for efficient to multiply tensors rather than vectors. 



## KroneckerSum

The [`KroneckerSum`](../../api/kroneckersum) class is used to represent the Kronecker sum of a chain of square matrices \(\mathbf{A}^{(i)}\). 
$$
\begin{align}
\mathbf{A} &= \mathbf{A}^{(1)} \oplus \mathbf{A}^{(2)} \oplus \;\dots \; \oplus \mathbf{A}^{(n)} \\[0.2cm]
&= \mathbf{A}^{(1)} \otimes \mathbf{I}^{(2)} \otimes \;\dots \; \otimes \mathbf{I}^{(n)} \; \dots \; + \; \dots \; \mathbf{I}^{(1)} \otimes \mathbf{I}^{(2)} \otimes \;\dots \; \otimes \mathbf{A}^{(n)}\\[0.2cm]
&= \bigoplus_{i=1}^n \mathbf{A}^{(i)}
\end{align}
$$
Just as with the [`KroneckerProduct`](../../api/kroneckerproduct) class, it is instantiated by passing a sequence of square arrays. 

```python
from pykronecker import KroneckerSum

KS = KroneckerSum(As)
```

Again, `KS` can now operate on tensors and vectors just as with [`KroneckerProduct`](../../api/kroneckerproduct)s. 

```python
assert (KS @ x).shape == (30 * 40 * 50, )
assert (KS @ X).shape == (30, 40, 50)
```



## KroneckerDiag

The [`KroneckerDiag`](../../api/kroneckerdiag) class is used to represent diagonal operators, where the diagonal is a tensor which has been vectorised in *row major* order. 
$$
\mathbf{A} = \text{diag}\big( \text{vec}_{\text{rm}}(\mathbf{D}) \big)
$$
Note, this is different from the \(\text{vec}(\cdot)\) operator [commonly used in mathematics](https://en.wikipedia.org/wiki/Vectorization_(mathematics)), which is defined in terms of column major vectorisation. We use row major vectorisation in this library for two reasons:

* It aligns better with [NumPy's memory organisation](https://numpy.org/devdocs/dev/internals.html). 
* It means Kronecker products created from matrices of size  \((N_1, N_2, ..., N_n)\)  act on tensors of shape \((N_1, N_2, ..., N_n)\) rather than shape \((N_n, ..., N_2, N_1)\), which is less confusing. 

`KroneckerDiag`s can be instantiated by passing an \(n\)-dimensional NumPy array. 

```python
from pykronecker import KroneckerDiag

D = np.random.normal(size=Ns)

KD = KroneckerDiag(D)
```

This can operate on tensors and vectors of the same shape as those operated on by [`KroneckerProduct`](../../api/kroneckerproduct)s and [`KroneckerSum`](../../api/kroneckersum)s. 

```python
assert (KD @ x).shape == (30 * 40 * 50, )
assert (KD @ X).shape == (30, 40, 50)
```



## KroneckerIdentity

The [`KroneckerIdentity`](../../api/kroneckeridentity) class is used to represent an identity matrix. It can be instantiated by passing either

* `like`: another [`KroneckerOperator`](../../api/kroneckeroperator) of the same size, or
* `tensor_shape`: the shape of tensors this operator should act on

```python
from pykronecker import KroneckerIdentity

KI1 = KroneckerIdentity(like=KS)
KI2 = KroneckerIdentity(tensor_shape=Ns)

assert (KI1 @ x).shape == (30 * 40 * 50, ) and (KI2 @ x).shape == (30 * 40 * 50, )
assert (KI1 @ X).shape == (30, 40, 50) and (KI2 @ X).shape == (30, 40, 50)
```

