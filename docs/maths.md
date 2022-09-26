# Maths

The result of taking the Kronecker product of two matrices \(\mathbf{A}^{(1)}\) and \(\mathbf{A}^{(2)}\), with shapes \( (N_1 \times N_1) \) and \((N_2 \times N_2)\) respectively, is a larger matrix of shape \((N_1 N_2 \times N_1 N_2 )\). It is denoted as \( \mathbf{A}^{(1)} \otimes \mathbf{A}^{(2)} \) and is given by 

$$
\mathbf{A}^{(1)} \otimes \mathbf{A}^{(2)} = 
\begin{bmatrix} 
\mathbf{A}_{11}^{(1)} \mathbf{A}^{(2)} & \dots  & \mathbf{A}_{1 N_1}^{(1)} \mathbf{A}^{(2)} \\
\vdots   & \ddots & \vdots   \\
\mathbf{A}_{N_1 1}^{(1)} \mathbf{A}^{(2)} & \dots  & \mathbf{A}_{N_1 N_1}^{(1)} \mathbf{A}^{(2)} 
\end{bmatrix}
$$

In general, multiple matrices can be chained together via the Kronecker product to produce larger and larger systems. 

$$
\mathbf{A} = \bigotimes_{i=1}^k \mathbf{A}^{(i)} \quad \text{shape:} \quad \big( N \times N \big), \quad \text{where} \quad N = \prod_{i=1}^k N_i
$$

Naively, multiplying this matrix onto a vector has time and memory complexity of \( O(N^2) \). However, this can be reduced to \(O(\sum N_i^2)\) and \(O(N \sum N_i^2)\) respectively by taking advantage of the mathematical properties of the Kronecker product. This makes many problems possible that would otherwise be intractable. 