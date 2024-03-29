# from __future__ import annotations
# from abc import abstractmethod, ABC
# from typing import Callable, List, Union
#
# import numpy as np
# from numpy import ndarray
#
#
#
# from pykronecker.base import KroneckerOperator
# from pykronecker.composite import OperatorSum
# from pykronecker.utils import numeric
#
# try:
#     import jax.numpy as jnp
# except ImportError:
#     import numpy as jnp
#
# class KroneckerBlockBase(KroneckerOperator, ABC):
#     """
#     Abstract class to capture shared behaviour of KroneckerBlock and KroneckerBlockDiag
#     """
#
#     def __init__(self, blocks: list):
#
#         self.check_blocks_consistent(blocks)
#         self.blocks = blocks
#         self.n_blocks = len(blocks)
#         self._block_sizes = [block.shape[0] for block in self.blocks]
#         self._cum_block_sizes = [0] + np.cumsum(self._block_sizes).tolist()
#         N = sum(self._block_sizes)
#         self.shape = (N, N)
#         self.tensor_shape = self.get_tensor_shape()
#
#     @abstractmethod
#     def _apply_to_blocks(self, function: Callable, transpose=False) -> list:
#         """
#         Apply function to each block in turn, then return the resultant blocks. If
#         `transpose`, return the blocks in their transposed order. Implemented by
#         subclasses.
#         """
#         raise NotImplementedError
#
#     @abstractmethod
#     def _get_block_sizes(self) -> list:
#         raise NotImplementedError
#
#     @abstractmethod
#     def _diag_blocks(self) -> list:
#         raise NotImplementedError
#
#     @property
#     def input_shape(self) -> tuple:
#         return tuple(
#             block.input_shape if isinstance(block, KroneckerOperator) else block.shape[1] for
#             block in self._diag_blocks())
#
#     @property
#     def output_shape(self) -> tuple:
#         return tuple(
#             block.output_shape if isinstance(block, KroneckerOperator) else block.shape[0] for
#             block in self._diag_blocks())
#
#     def iter_edges(self):
#         """
#         Return generator expression that iterates through the indices representing
#         the left and right boundaries of each diagonal block
#         """
#         return zip(self._cum_block_sizes[:-1], self._cum_block_sizes[1:])
#
#     @staticmethod
#     def check_blocks_consistent(blocks: list):
#         """
#         Check the blocks, which are provided as input to KroneckerBlock and KroneckerBlockDiag are consistent
#         """
#
#         def replace(items):
#             """
#             Turn into a block of stings, so we can use numpy.block
#             """
#             out = []
#             for item in items:
#                 if isinstance(item, (list, tuple)):
#                     out.append(replace(item))
#                 else:
#                     out.append('a')
#             return out
#
#         ndim = np.block(replace(blocks)).ndim
#
#         if ndim == 1:
#             assert all(hasattr(block, 'shape') for block in blocks)
#             assert all(block.shape[0] == block.shape[1] for block in blocks)
#
#         elif ndim == 2:
#
#             # check diagonal blocks are square
#             assert all(blocks[i][i].shape[0] == blocks[i][i].shape[1] for i in range(len(blocks)))
#             shapes = [blocks[i][i].shape[0] for i in range(len(blocks))]
#
#             for i in range(len(blocks)):
#                 for j in range(len(blocks)):
#                     assert hasattr(blocks[i][j], 'shape')
#                     assert blocks[i][j].shape == (shapes[i], shapes[j])
#
#         else:
#             raise ValueError(f'blocks should be 1d or 2d but it is {np.ndim(blocks)}d')
#
#         return True
#
#     @property
#     def T(self) -> 'KroneckerBlockBase':
#         return self.factor * self.__class__(blocks=self._apply_to_blocks(lambda block: block.T, transpose=True))
#
#     def conj(self) -> 'KroneckerBlockBase':
#         return np.conj(self.factor) * self.__class__(blocks=self._apply_to_blocks(lambda block: block.conj(), transpose=False))
#
#     def __pow__(self, power: numeric, modulo=None) -> 'KroneckerBlockBase':
#         new = self.__class__(blocks=self._apply_to_blocks(lambda block: block ** power))
#         new.factor = self.factor ** power
#         return new
#
#     def __copy__(self) -> 'KroneckerBlockBase':
#         new = self.__class__(blocks=self._apply_to_blocks(lambda block: block.copy() if isinstance(block, KroneckerOperator) else block))
#         new.factor = self.factor
#         return new
#
#     def __deepcopy__(self, memodict=None) -> 'KroneckerBlockBase':
#         new = self.__class__(blocks=self._apply_to_blocks(lambda block: block.deepcopy() if isinstance(block, KroneckerOperator) else block.copy()))
#         new.factor = self.factor
#         return new
#
#
# class KroneckerBlock(KroneckerBlockBase):
#
#     def __init__(self, blocks: List[list]):
#         """
#         Create a general block operator. Items in the block can be arrays or operators.
#
#         E.g. blocks = [[A11, A12, A13]
#                        [A21, A22, A23]
#                        [A31, A32, A33]]
#         """
#         super().__init__(blocks)
#         self.dtype = np.result_type(*[self.blocks[i][i].dtype for i in range(len(self.blocks))])
#
#     def _apply_to_blocks(self, function: Callable, transpose=False):
#         """
#         Helper method: apply `function` to each block, and return as nested list
#         """
#
#         if transpose:
#             return [[function(self.blocks[j][i]) for j in range(self.n_blocks)] for i in range(self.n_blocks)]
#         else:
#             return [[function(self.blocks[i][j]) for j in range(self.n_blocks)] for i in range(self.n_blocks)]
#
#     def _get_block_sizes(self) -> list:
#         return [block.shape[0] for block in self.blocks]
#
#     def _diag_blocks(self):
#         return [self.blocks[i][i] for i in range(len(self.blocks))]
#
#     def operate(self, other: ndarray) -> ndarray:
#
#         x = other.squeeze()
#
#         if x.shape[0] != len(self):
#             raise ValueError(f'other\'s first dimension should have length {len(self)} to match the dimensions of the operator, but it has length {x.shape[0]}')
#
#         if x.ndim not in [1, 2]:
#             raise ValueError(f'other must be 1 or 2d but it is {other.ndim}d')
#
#         return self.factor * jnp.concatenate([sum(self.blocks[i][j] @ x[n1:n2] for j, (n1, n2) in enumerate(self.iter_edges())) for i in range(self.n_blocks)], axis=0)
#
#     def __mul__(self, other: Union['KroneckerOperator', numeric]) -> KroneckerOperator:
#
#         if isinstance(other, KroneckerOperator):
#
#             self.check_operators_consistent(self, other)
#
#             if isinstance(other, KroneckerBlock):
#                 new_blocks = [[self.blocks[i][j] * other.blocks[i][j] for j in range(self.n_blocks)] for i in range(self.n_blocks)]
#                 return self.factor * other.factor * KroneckerBlock(new_blocks)
#
#             elif isinstance(other, KroneckerBlockDiag):
#                 new_blocks = [self.blocks[i][i] * other.blocks[i] for i in range(self.n_blocks)]
#                 return self.factor * other.factor * KroneckerBlockDiag(new_blocks)
#
#             elif isinstance(other, OperatorSum):
#                 return other.factor * OperatorSum(self * other.A, self * other.B)
#
#             else:
#                 raise TypeError('A KroneckerBlock cannot be multiplied element-wise onto this operator')
#
#         # otherwise other should be a scalar, handled in the base class
#         else:
#             return super().__mul__(other)
#
#     def inv(self) -> 'KroneckerBlock':
#         raise NotImplementedError
#
#     def to_array(self) -> ndarray:
#         return self.factor * np.block(self._apply_to_blocks(lambda block: block.to_array() if isinstance(block, KroneckerOperator) else block))
#
#     def diag(self):
#
#         def get_diag(block: KroneckerOperator | ndarray):
#             return block.diag() if isinstance(block, KroneckerOperator) else np.diag(block)
#
#         return self.factor * np.concatenate([get_diag(self.blocks[i][i]) for i in range(self.n_blocks)])
#
#     def __repr__(self) -> str:
#
#         def to_string(block):
#             return str(block) if isinstance(block, KroneckerOperator) else f'ndarray({block.shape})'
#
#         return 'KroneckerBlock([{}])'.format(', '.join(['[' + ', '.join([to_string(self.blocks[i][j]) for j in range(self.n_blocks)]) + ']' for i in range(self.n_blocks)]))
#
#
# class KroneckerBlockDiag(KroneckerBlockBase):
#
#     def __init__(self, blocks: list):
#         """
#         Create a diagonal block operator. Items in the block can be arrays or operators.md.
#
#         E.g. blocks = [A1, A2, A3] -> [[A1, 0, 0]
#                                        [0, A2, 0]
#                                        [0, 0, A3]]
#         """
#         super().__init__(blocks)
#         self.dtype = np.result_type(*[self.blocks[i].dtype for i in range(len(self.blocks))])
#
#     def _apply_to_blocks(self, function: Callable, transpose=False):
#         return [function(self.blocks[i]) for i in range(self.n_blocks)]
#
#     def _diag_blocks(self):
#         return [self.blocks[i] for i in range(len(self.blocks))]
#
#     def operate(self, other: ndarray) -> ndarray:
#         """
#         Other should be a vector or a matrix of vector columns
#         """
#
#         x = other.squeeze()
#
#         if x.shape[0] != len(self):
#             raise ValueError(f'other\'s first dimension should have length {len(self)} to match the dimensions of the operator, but it has length {x.shape[0]}')
#
#         if x.ndim == 1:
#             return self.factor * np.concatenate([block @ x[n1:n2] for block, (n1, n2) in zip(self.blocks, self.iter_edges())], axis=0)
#
#         elif x.ndim == 2:
#             return self.factor * np.concatenate([block @ x[n1:n2, :] for block, (n1, n2) in zip(self.blocks, self.iter_edges())], axis=0)
#
#         else:
#             raise ValueError('other must be 1 or 2d')
#
#     def __mul__(self, other: Union['KroneckerOperator', numeric]) -> KroneckerOperator:
#
#         if isinstance(other, KroneckerOperator):
#
#             self.check_operators_consistent(self, other)
#
#             if isinstance(other, KroneckerBlock):
#                 new_blocks = [self.blocks[i] * other.blocks[i][i] for i in range(self.n_blocks)]
#                 return self.factor * other.factor * KroneckerBlockDiag(new_blocks)
#
#             elif isinstance(other, KroneckerBlockDiag):
#                 new_blocks = [self.blocks[i] * other.blocks[i] for i in range(self.n_blocks)]
#                 return self.factor * other.factor * KroneckerBlockDiag(new_blocks)
#
#             elif isinstance(other, OperatorSum):
#                 return other.factor * OperatorSum(self * other.A, self * other.B)
#
#             else:
#                 raise TypeError('A KroneckerBlock cannot be multiplied element-wise onto this operator')
#
#         # otherwise other should be a scalar, handled in the base class
#         else:
#             return super().__mul__(other)
#
#     def inv(self) -> 'KroneckerBlockDiag':
#         return self.factor ** -1 * KroneckerBlockDiag(blocks=self._apply_to_blocks(lambda block: block.inv() if isinstance(block, KroneckerOperator) else np.linalg.inv(block)))
#
#     def to_array(self) -> ndarray:
#
#         out = np.zeros(self.shape, dtype=self.dtype)
#
#         for block, (n1, n2) in zip(self.blocks, self.iter_edges()):
#
#             if isinstance(block, KroneckerOperator):
#                 out[n1:n2, n1:n2] = block.to_array()
#             else:
#                 out[n1:n2, n1:n2] = block
#
#         return self.factor * out
#
#     def diag(self) -> ndarray:
#         return self.factor * np.concatenate(self._apply_to_blocks(lambda block: block.diag() if isinstance(block, KroneckerOperator) else np.diag(block)))
#
#     def __repr__(self) -> str:
#         return 'KroneckerBlockDiag([{}])'.format(', '.join([str(block) if isinstance(block, KroneckerOperator) else f'ndarray{block.shape}' for block in self.blocks]))
