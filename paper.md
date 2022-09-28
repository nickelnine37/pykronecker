---
title: 'PyKronecker: A Python Library for the Efficient Manipulation of Kronecker Products and Related Structures'
tags:
  - Python
  - Numpy
  - Jax
  - Kronecker product
  - Kronecker sum
  - linear system
  - GPU
authors:
  - name: Edward Antonian
    affiliation: 1
  - name: Gareth W. Peters
    affiliation: 2
  - name: Michael Chantler
    affiliation: 1
affiliations:
 - name: Heriot-Watt University, UK
   index: 1
 - name: University of California Santa Barbara, USA
   index: 2
date: 27 September 2022
bibliography: paper.bib
---

# Summary

Matrix operators composed of Kronecker products and related objects, such as Kronecker sums, arise in many areas of applied mathematics including signal processing, semidefinite programming, and quantum computing [@loan2000]. As such, a computational toolkit for manipulating Kronecker-based systems, in a way that is both efficient and idiomatic, has the potential to aid research in many fields.  PyKronecker aims to deliver this in the Python programming language by providing a simple API that integrates well with the widely-used NumPy library [@harris2020], with support for automatic differentiation and accelerated computation on GPU/TPU hardware using Jax [@jax2018].  

The Kronecker product of an $(n \times n)$ matrix $\mathbf{A}$ and an $(m \times m)$ matrix $\mathbf{B}$, denoted $\mathbf{A} \otimes \mathbf{B}$, is defined  by
$$
\mathbf{A} \otimes \mathbf{B} = 
\begin{bmatrix} 
\mathbf{A}_{1,1} \mathbf{B} & \dots  & \mathbf{A}_{1,n} \mathbf{B} \\
\vdots   & \ddots & \vdots   \\
\mathbf{A}_{n,1} \mathbf{B} & \dots  & \mathbf{A}_{n,n} \mathbf{B}
\end{bmatrix}
$$

The resultant operator has shape $(nm \times nm)$ and, as such, can act on vectors of length $nm$. The Kronecker sum of $\mathbf{A}$ and $\mathbf{B}$, denoted $\mathbf{A} \oplus \mathbf{B}$ can be defined in terms of the Kronecker product as 
$$
\mathbf{A} \oplus \mathbf{B} = \mathbf{A} \otimes \mathbf{I}_m + \mathbf{I}_n \otimes \mathbf{B}
$$
where $\mathbf{I}_d$ is the $d$-dimensional identity matrix, resulting in an operator of the same size as $\mathbf{A} \otimes \mathbf{B}$. By applying these definitions recursively, the Kronecker product or sum of more than two matrices can also be defined. In general, the Kronecker product/sum of $k$ square matrices $\{ \mathbf{A}^{(i)} \}_{i=1}^k$, with shapes $\{n_i \times n_i\}_{i=1}^k$ can be written respectively as
$$
\bigotimes_{i=1}^k \mathbf{A}^{(i)} = \mathbf{A}^{(1)} \otimes \mathbf{A}^{(2)} \otimes \dots \otimes \mathbf{A}^{(k)}
$$
and 
$$
\bigoplus_{i=1}^k \mathbf{A}^{(i)} = \mathbf{A}^{(1)} \oplus \mathbf{A}^{(2)} \oplus \dots \oplus \mathbf{A}^{(k)}
$$
The resultant operators can act on either vectors of length $N = \prod_{i=1}^k n_i$, or equivalently tensors of shape $(n_1, n_2, \dots n_k)$. Whilst a naive implementation of matrix-vector multiplication in this space has time and memory complexity of $O(N^2)$ a more efficient implementation can be achieved using the 'generalized vec trick' [@Airola2018]. By applying this algorithm, the required memory and time complexity is reduced to $O(\sum_{i=1}^k n_i^2)$, and $O(N \sum_{i=1}^k n_i)$ respectively, making it possible to solve many problems that would otherwise be intractable. 

# Statement of need

PyKronecker is aimed at researchers in any area of applied mathematics where systems involving Kronecker products arise. It has been designed with the following specific goals in mind.

a) *To provide a simple and intuitive object-oriented interface for manipulating systems involving Kronecker-products with Python*.

In PyKronecker, users can define new composite operators by applying familiar matrix operations such as scaling, addition, matrix multiplication and transposition. This allows the user to manipulate Kronecker operators as if they are large NumPy arrays, rather than having to write efficient but sometimes cryptic expressions involving the individual sub-matrices. This can greatly simplify code, making it easier to read, debug and refactor.  

b) *To execute matrix-vector multiplications in a way that is maximally efficient and runs on parallel GPU/TPU hardware.*

By making use of the generalized vec trick, Just In Time (JIT) compilation, and parallel processing, PyKronecker can achieve very fast execution times compared to alternative implementations. This complexity is handled under the hood, allowing users to focus on their research goals without concerning themselves with performance. 

c) *To allow automatic differentiation for complex loss functions involving Kronecker products.*

Many widely-used optimisation algorithms in Machine Learning (ML), such as stochastic gradient descent, rely on rapidly evaluating the derivative of an objective function. Automatic differentiation has played a key role in accelerating ML research by removing the need to manually derive analytical gradients [@baydin2018]. By integrating with the Jax library, PyKronecker enables automatic differentiation of complex functions involving Kronecker products out of the box. 

To the best of the our knowledge, no existing software achieves all three of these aims. 

## Comparison with other libraries

One potential alternative in Python is the PyLops library which provides an interface for general functionally-defined linear operators, and includes a Kronecker product implementation [@Ravasi2020]. It also supports GPU acceleration with CuPy [@okuta2017]. However, as a more general library, PyLops does not provide support for the Kronecker product of more than two matrices, implement a Kronecker sum operator, implement matrix-tensor multiplication, or provide automatic differentiation. It is also significantly slower than PyKronecker when operating on simple NumPy arrays. 

Another alternative is the library Kronecker.jl [@Stock2020], implemented in the Julia programming language [@bezanson2017]. Kronecker.jl has many of the same aims as PyKronecker and has a a clean interface, making use of Julia's unicode support to create Kronecker products with a custom `⊗` operator. However, at this time, the library does not support GPU acceleration or automatic differentiation, although the former is in development. 

Table 1. shows a feature comparison of these libraries, along with a custom NumPy implementation using  the vec trick. It also shows the time to compute the multiplication of a Kronecker product against a vector for two scenarios. In the first, the Kronecker product is constructed from two of matrices of size $(400 \times 400)$ and $(500 \times 500)$, and in the second Kronecker product is constructed from three of matrices of size $(100 \times 100)$,  $(150 \times 150)$ and  $(200 \times 200)$ respectively. Experiments were performed with an Intel Core  2.80GHz i7-7700HQ CPU, and an Nvidia 1050Ti GPU.  In both cases, PyKronecker on the GPU is the fastest by a significant margin. 

| Implementation    | Python | Autodiff | GPU support | Compute time (400, 500) | Compute time (100, 150, 200) |
| ----------------- | ------ | -------- | ----------- | ----------------------- | ---------------------------- |
| Pure NumPy        | Yes    | No       | No          | 5.04 ms ± 343 µs        | 38.9 ms ± 4.07 ms            |
| Kronecker.jl      | No     | No       | No          | 9.61 ms ± 881 µs        | 380 ms ± 6.15 ms             |
| PyLops (CPU)      | Yes    | No       | No          | 17.9 ms ± 986 µs        | 478 ms ± 4.79 ms             |
| PyLops (GPU)      | Yes    | No       | Yes         | 54.6 ms ± 1.04 ms       | 4.06 s ± 182 ms              |
| PyKronecker (CPU) | Yes    | Yes      | No          | 4.74 ms ± 318 µs        | 15.1 ms ± 2.24 ms            |
| PyKronecker (GPU) | Yes    | Yes      | Yes         | 261 µs ± 17.3 µs        | 258 µs ± 78.2 µs             |

# Outlook and Future Work

There are several features that we are developing to expand the functionality of PyKronecker. The first is to provide support for non-square operators. In a typical problem, the Kronecker operators encountered are square, however, there are a significant minority of contexts where this is not the case. 

* Solvers
* Sparse matrices

# References