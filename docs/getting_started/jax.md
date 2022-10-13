# Acceleration with Jax

PyKronecker is also compatible with [Jax](https://jax.readthedocs.io/en/latest/) arrays for accelerated execution. However, since Jaxlib cannot currently be installed on Windows, this feature is available only on Linux and MacOS. 

The main benefit of using Jax is the Just In Time (JIT) compiler, which significantly decreases the execution time for matrix-vector multiplications. In addition, Jax supports [automatic differentiation](https://pykronecker.readthedocs.io/en/latest/getting_started/autodiff/).  With Jax installed, PyKronecker can also be run on systems with a compatible Nvidia GPU/TPU. Note that Jax does not bundle CUDA or CuDNN as part of the `pip` package, and as such these must be installed separately. 


The figure below compares some of the compute times for the vanilla version of PyKronecker, PyKronecker with Jax's JIT, and PyKronecker with Jax + a 1050TI GPU. 

![](https://raw.githubusercontent.com/nickelnine37/pykronecker/main/docs/img/test.svg)



## Using Jax arrays with PyKronecker

Once you have Jax installed into your environment, Jax arrays can be used more or less in place of NumPy arrays. 

```python 
import jax
import jax.numpy as jnp
from pykronecker import KroneckerProduct

Ns = (30, 40, 50)

As = [jnp.asarray(np.random.normal(size=(N, N))) for N in Ns]
X = jnp.asarray(np.random.normal(size=Ns))

KP = KroneckerProduct(As)

print(KP @ X)
```

