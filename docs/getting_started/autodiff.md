# Automatic differentiation

Automatic differentiation of functions involving Kronecker operators, which map tensors or vectors to a real number, can be achieved via Jax's [`grad`](https://jax.readthedocs.io/en/latest/_autosummary/jax.grad.html) function. This requires that Jax be installed and, as such, is only available on Linux and MacOS operating systems. 


Consider the following function 

$$
f(x) = \mathbf{x}^\top \mathbf{A} \mathbf{x} 
$$

where \(\mathbf{x}\) is a vector, and \(\mathbf{A}\) is a matrix. It is well-known that the gradient of this function is given by 

$$
\nabla f(x) = (\mathbf{A} + \mathbf{A}^\top) \mathbf{x} 
$$

We can automatically calculate this gradient, even when \(\mathbf{A}\) is a complex Kronecker operator. For example:

```python
from pykronecker import KroneckerProduct, KroneckerSum, KroneckerDiag
import numpy as np
import jax.numpy as jnp
from jax import grad

Ns = [30, 40, 50]
As = [jnp.asarray(np.random.normal(size=(Ni, Ni))) for Ni in Ns]
Bs = [jnp.asarray(np.random.normal(size=(Ni, Ni))) for Ni in Ns]
D = jnp.asarray(np.random.normal(size=Ns))

A = KroneckerProduct(As) @ KroneckerSum(Bs) + KroneckerDiag(D)


def f(x):
    return x.T @ A @ x


df = grad(f)

x = jnp.asarray(np.random.normal(size=np.prod(Ns)))

assert np.allclose((A + A.T) @ x, df(x), rtol=1e-2, atol=1e-4)
```