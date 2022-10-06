# Installing PyKronecker

PyKronecker is available for Python >= 3.7, and can be installed on Windows, MacOS and Linux by running

```bash
pip3 install pykronecker
```

This will install the vanilla NumPy version of the library. This means [`KroneckerOperator`](../api/kroneckeroperator)s can be constructed from, and operate on, NumPy arrays only. 

Linux and MacOS users have the additional option of installing PyKronecker with [Jax](https://jax.readthedocs.io/en/latest/index.html) support. The benefit of this is significantly faster runtimes, even when working with NumPy arrays only, due to Jax's JIT complier. This can be installed by running

```bash
pip3 install pykronecker[jax]
```

Note that jaxlib is only officially supported on Linux (Ubuntu 16.04 or later) and macOS (10.12 or later) platforms.

For Linux users with an Nvidia graphics card, PyKronecker is also compatible with the GPU and TPU version of Jax. However, this Jax version relies on [CUDA Toolkit](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html) and [cuDNN](https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html) which can be trickier to install. For this reason, it is recommended to follow the instructions [here](https://github.com/google/jax#installation) to install the GPU-compatible version of Jax prior to installing PyKronecker. 

The figure below shows the compute time for multiplying a [`KroneckerProduct`](../api/kroneckerproduct), [`KroneckerSum`](../api/kroneckersum) and [`KroneckerDiag`](../api/kroneckerdiag) onto a tensor of shape `(100, 120, 140)` using NumPy arrays only, NumPy arrays with the Jax JIT compiler, and Jax arrays on an Nvidia 1050Ti GPU.   

![](https://raw.githubusercontent.com/nickelnine37/pykronecker/main/docs/img/test.svg)
