from __future__ import annotations
from abc import abstractmethod, ABC
from typing import Callable, List

import numpy as np
from numpy import ndarray

from pykronecker.base import KroneckerOperator
from pykronecker.types import numeric




class KroneckerBlockBase(KroneckerOperator, ABC):
    """
    Abstract class to capture shared behaviour of KroneckerBlock and KroneckerBlockDiag
    """

    def __init__(self, blocks: list):

        self.check_blocks_consistent(blocks)
        self.blocks = blocks
        self.n_blocks = len(blocks)
        self.block_sizes = self.get_block_sizes()
        self.cum_block_sizes = [0] + np.cumsum(self.block_sizes).tolist()
        N = sum(self.block_sizes)
        self.shape = (N, N)

    @abstractmethod
    def apply_to_blocks(self, function: Callable, transpose=False) -> list:
        """
        Apply function to each block in turn, then return the resultant blocks. If
        `transpose`, return the blocks in their transposed order. Implemented by
        subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def get_block_sizes(self):
        raise NotImplementedError
    
    def iter_edges(self):
        """
        Return generator expression that iterates through the indices represesenting
        the left and right boundaries of each diagonal block
        """
        return zip(self.cum_block_sizes[:-1], self.cum_block_sizes[1:])

    @staticmethod
    def check_blocks_consistent(blocks: list):
        """
        Check the blocks, which are provided as input to KroneckerBlock and KroneckerBlockDiag are consistent
        """

        ndim = np.asarray(blocks, dtype='object').ndim

        if ndim == 1:
            assert all(isinstance(block, (KroneckerOperator, ndarray)) for block in blocks)
            assert all(block.shape[0] == block.shape[1] for block in blocks)

        elif ndim == 2:

            # check diagonal blocks are square
            assert all(blocks[i][i].shape[0] == blocks[i][i].shape[1] for i in range(len(blocks)))
            shapes = [blocks[i][i].shape[0] for i in range(len(blocks))]

            for i in range(len(blocks)):
                for j in range(len(blocks)):
                    assert isinstance(blocks[i][j], (KroneckerOperator, ndarray))
                    assert blocks[i][j].shape == (shapes[i], shapes[j])

        else:
            raise ValueError(f'blocks should be 1d or 2d but it is {np.ndim(blocks)}d')

        return True

    @property
    def T(self) -> 'KroneckerBlockBase':
        return self.factor * self.__class__(blocks=self.apply_to_blocks(lambda block: block.T, transpose=True))

    def __pow__(self, power: numeric, modulo=None) -> 'KroneckerBlockBase':
        new = self.__class__(blocks=self.apply_to_blocks(lambda block: block ** power))
        new.factor = self.factor ** power
        return new

    def __copy__(self) -> 'KroneckerBlockBase':
        new = self.__class__(blocks=self.apply_to_blocks(lambda block: block.copy() if isinstance(block, KroneckerOperator) else block))
        new.factor = self.factor
        return new

    def __deepcopy__(self, memodict={}) -> 'KroneckerBlockBase':
        new = self.__class__(blocks=self.apply_to_blocks(lambda block: block.deepcopy() if isinstance(block, KroneckerOperator) else block.copy()))
        new.factor = self.factor
        return new


class KroneckerBlock(KroneckerBlockBase):

    def __init__(self, blocks: List[list]):
        """
        Create a general block operator. Items in the block can be arrays or operators.

        E.g. blocks = [[A11, A12, A13]
                       [A21, A22, A23]
                       [A31, A32, A33]]
        """
        super().__init__(blocks)


    def apply_to_blocks(self, function: Callable, transpose=False):
        """
        Helper method: apply `function` to each block, and return as nested list
        """

        if transpose:
            return [[function(self.blocks[j][i]) for j in range(self.n_blocks)] for i in range(self.n_blocks)]
        else:
            return [[function(self.blocks[i][j]) for j in range(self.n_blocks)] for i in range(self.n_blocks)]

    def get_block_sizes(self):
        return [self.blocks[i][i].shape[0] for i in range(len(self.blocks))]

    def operate(self, other: ndarray) -> ndarray:

        if other.ndim == 1:

            assert len(other) == self.shape[1]

            out = [np.zeros_like(other[n1:n2]) for n1, n2 in self.iter_edges()]
            other = [other[n1:n2] for n1, n2 in self.iter_edges()]

        elif other.ndim == 2:

            assert other.shape[0] == self.shape[1]

            out = [np.zeros_like(other[n1:n2, :]) for n1, n2 in self.iter_edges()]
            other = [other[n1:n2, :] for n1, n2 in self.iter_edges()]

        else:
            raise ValueError(f'other must be 1 or 2d but it is {other.ndim}d')

        for i in range(self.n_blocks):
            for j in range(self.n_blocks):
                out[i] += self.blocks[i][j] @ other[j]

        return self.factor * np.concatenate(out, axis=0)

    def inv(self) -> 'KroneckerBlock':
        raise NotImplementedError

    def to_array(self) -> ndarray:
        return self.factor * np.block(self.apply_to_blocks(lambda block: block.to_array() if isinstance(block, KroneckerOperator) else block))

    def __repr__(self) -> str:

        def to_string(block):
            return str(block) if isinstance(block, KroneckerOperator) else f'ndarray({block.shape})'

        return 'KroneckerBlock([{}])'.format(', '.join(['[' + ', '.join([to_string(self.blocks[i][j]) for j in range(self.n_blocks)]) + ']' for i in range(self.n_blocks)]))


class KroneckerBlockDiag(KroneckerBlockBase):

    def __init__(self, blocks: list):
        """
        Create a diagonal block operator. Items in the block can be arrays or operators.

        E.g. blocks = [A1, A2, A3] -> [[A1, 0, 0]
                                       [0, A2, 0]
                                       [0, 0, A3]]
        """
        super().__init__(blocks)

    def apply_to_blocks(self, function: Callable, transpose=False):
        return [function(self.blocks[i]) for i in range(self.n_blocks)]

    def get_block_sizes(self):
        return [self.blocks[i].shape[0] for i in range(len(self.blocks))]

    def operate(self, other: ndarray) -> ndarray:
        """
        other should be a vector only
        """

        if other.ndim == 1:
            assert len(other) == self.shape[1]
            return self.factor * np.concatenate([block @ other[n1:n2] for block, (n1, n2) in zip(self.blocks, self.iter_edges())], axis=0)

        elif other.ndim == 2:
            assert other.shape[0] == self.shape[1]
            return self.factor * np.concatenate([block @ other[n1:n2, :] for block, (n1, n2) in zip(self.blocks, self.iter_edges())], axis=0)

        else:
            raise ValueError('other must be 1 or 2d')

    def inv(self) -> 'KroneckerBlockDiag':
        return self.factor ** -1 * KroneckerBlockDiag(blocks=[block.inv() if isinstance(block, KroneckerOperator) else np.linalg.inv(block) for block in self.blocks])

    def to_array(self) -> ndarray:

        out = np.zeros(self.shape)

        for block, (n1, n2) in zip(self.blocks, self.iter_edges()):

            if isinstance(block, KroneckerOperator):
                out[n1:n2, n1:n2] = block.to_array()
            else:
                out[n1:n2, n1:n2] = block

        return self.factor * out

    def __repr__(self) -> str:
        return 'KroneckerBlockDiag([{}])'.format(', '.join([str(block) if isinstance(block, KroneckerOperator) else f'ndarray{block.shape}' for block in self.blocks]))
