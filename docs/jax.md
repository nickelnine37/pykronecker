# Acceleration with Jax

PyKronecker is also compatible with [Jax](https://jax.readthedocs.io/en/latest/) arrays for accelerated execution. However, since Jax is currently only available for installation on Linux, this feature is not available on other operating systems. Linux users can install the jax-compatible version of PyKronecker by running 

```bash
pip install pykronecker[jax]
```

The main benefit of using Jax is the Just In Time (JIT) compiler, which significantly decreases the execution time for matrix-vector multiplications. 

With Jax installed, PyKronecker can also be run on systems with a compatible Nvidia GPU/TPU. Note that Jax does not bundle CUDA or CuDNN as part of the `pip` package, and as such these must be installed separately. The easiest way to do this is via `conda` with 

```bash
conda install cuda -c nvidia
```

and 

```bash
conda install -c anaconda cudnn
```

Next, Jax can be installed with 

```bash
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

The figure below compares some of the compute times for the vanilla version of PyKronecker, PyKronecker with Jax's JIT, and PyKronecker with Jax + a 1050TI GPU. 

![](https://raw.githubusercontent.com/nickelnine37/pykronecker/main/docs/img/test.svg)



## Using Jax arrays with PyKronecker

Once you have Jax installed into your environment, Jax arrays can be used more or less in place of NumPy arrays. 

```python 
import jax
import jax.numpy as jnp
from pykronecker import KroneckerProduct

Ns = (30, 40, 50)

key = jax.random.PRNGKey(0)
As = [jax.random.normal(key, (N, N), dtype=jnp.float32) for N in Ns]
X = jax.random.normal(key, Ns, dtype=jnp.float32)

KP = KroneckerProduct(As)

print(KP @ X)
```

