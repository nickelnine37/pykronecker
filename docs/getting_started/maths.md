# A breif overview of Kronecker products

The Kronecker product of an \((n \times n)\) matrix \(\mathbf{A}\) and an \((m \times m)\) matrix \(\mathbf{B}\), denoted \(\mathbf{A} \otimes \mathbf{B}\), is defined  by
$$
\mathbf{A} \otimes \mathbf{B} = 
\begin{bmatrix} 
\mathbf{A}_{1,1} \mathbf{B} & \dots  & \mathbf{A}_{1,n} \mathbf{B} \\
\vdots   & \ddots & \vdots   \\
\mathbf{A}_{n,1} \mathbf{B} & \dots  & \mathbf{A}_{n,n} \mathbf{B}
\end{bmatrix}
$$

The resultant operator has shape \((nm \times nm)\) and, as such, can act on vectors of length \(nm\). The Kronecker sum of \(\mathbf{A}\) and \(\mathbf{B}\), denoted \(\mathbf{A} \oplus \mathbf{B}\) can be defined in terms of the Kronecker product as 
$$
\mathbf{A} \oplus \mathbf{B} = \mathbf{A} \otimes \mathbf{I}_m + \mathbf{I}_n \otimes \mathbf{B}
$$

where \(\mathbf{I}_d\) is the \(d\)-dimensional identity matrix, resulting in an operator of the same size as \(\mathbf{A} \otimes \mathbf{B}\). By applying these definitions recursively, the Kronecker product or sum of more than two matrices can also be defined. In general, the Kronecker product/sum of \(k\) square matrices \(\{ \mathbf{A}^{(i)} \}_{i=1}^k\), with shapes \(\{n_i \times n_i\}_{i=1}^k\) can be written respectively as

$$
\bigotimes_{i=1}^k \mathbf{A}^{(i)} = \mathbf{A}^{(1)} \otimes \mathbf{A}^{(2)} \otimes \dots \otimes \mathbf{A}^{(k)}
$$

and 

$$
\bigoplus_{i=1}^k \mathbf{A}^{(i)} = \mathbf{A}^{(1)} \oplus \mathbf{A}^{(2)} \oplus \dots \oplus \mathbf{A}^{(k)}
$$

The resultant operators can act on either vectors of length \(N = \prod_{i=1}^k n_i\) or equivalently tensors of shape \((n_1, n_2, \dots n_k)\). 

## 