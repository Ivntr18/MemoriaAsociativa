#!/usr/bin/env python

# Created by Raul Peralta-Lozada
# GPU version added by Ivan Torres-Rodriguez
import cupy
import numpy


class AssociativeMemoryError(Exception):
    pass


class AssociativeMemory(object):
    def __init__(self, n: int, m: int):
        """
        Parameters
        ----------
        n : int
            The size of the domain.
        m : int
            The size of the range.
        """
        self.n = n
        self.m = m
        self.grid = cupy.zeros((self.m, self.n), dtype=cupy.bool)

    def __str__(self):
        grid = cupy.zeros(self.grid.shape, dtype=cupy.unicode)
        grid[:] = 'O'
        r, c = cupy.nonzero(self.grid)
        for i in zip(r, c):
            grid[i] = 'X'
        return str(grid)

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, value: int):
        if value > 0:
            self._n = value
        else:
            raise ValueError('Invalid value for n.')

    @property
    def m(self):
        return self._m

    @m.setter
    def m(self, value: int):
        if value > 0:
            self._m = value
        else:
            raise ValueError('Invalid value for m.')

    @property
    def grid(self):
        return self._grid

    @grid.setter
    def grid(self, new_grid: cupy.ndarray):
        if (isinstance(new_grid, cupy.ndarray) and
                new_grid.dtype == cupy.bool and
                new_grid.shape == (self.m, self.n)):
            self._grid = new_grid
        else:
            raise ValueError('Invalid grid assignment.')

    @property
    def entropy(self) -> float:
        """Return the entropy of the Associative Memory."""
        e = 0.0  # entropy
        v = self.grid.sum(axis=0)  # number of marked cells in the columns
        for vi in v:
            if vi != 0:
                e += cupy.log2(1. / vi)
        e *= (-1.0 / self.n)
        return e

    @classmethod
    def from_grid(cls, grid: cupy.ndarray) -> 'AssociativeMemory':
        associative_mem = cls(grid.shape[1], grid.shape[0])
        associative_mem.grid = grid
        return associative_mem

    @staticmethod
    def vector_to_grid(vector, input_range, min_value):
        # now is only binary
        vector = cupy.ravel(vector)
        n = vector.size
        if vector.max() > input_range or vector.min() < min_value:
            raise ValueError('Values in the input vector are invalid. ',input_range,vector.max(),vector.min())
        grid = cupy.zeros((input_range, n), cupy.bool)
        vector -= min_value
        grid[vector, cupy.arange(vector.shape[0])] = True
        grid = cupy.flipud(grid)
        return grid
    
    @staticmethod
    def vector_to_grid_pad(vector, input_range, min_value, padding=1):
        # now is only binary
        vector = cupy.ravel(vector)
        n = vector.size
        if vector.max() > input_range or vector.min() < min_value:
            raise ValueError('Values in the input vector are invalid. ',input_range,vector.max(),vector.min())
        grid = cupy.zeros((input_range, n), cupy.bool)
        vector -= min_value
        for offset in range(-padding,padding+1):
            for vecOff,i in zip(vector+offset,cupy.arange(vector.shape[0])):
                if vecOff <= input_range and vecOff >= min_value: 
                    grid[int(vecOff), i] = True
        grid = cupy.flipud(grid)
        return grid
    
    
    @staticmethod
    def impl(a,b):
        return cupy.logical_or(cupy.logical_not(a),b)
    
    

    def abstract(self, vector_input, input_range=2, min_value=0) -> None:
        if vector_input.size != self.n:
            raise ValueError('Invalid size of the input data.')
        else:
            grid_input = self.vector_to_grid_pad(vector_input, input_range,
                                             min_value, int(numpy.round(0.1*input_range)) )
            #self.grid = self.grid | grid_input
            self.grid = cupy.logical_or(self.grid,grid_input)
            
            

    def reduce(self, vector_input, input_range=2, min_value=0):
        if vector_input.size != self.n:
            raise AssociativeMemoryError('Invalid size of the input data.')
        else:
            grid_input = self.vector_to_grid(vector_input,
                                             input_range, min_value)
            #grid_output = cupy.zeros(self.grid.shape, dtype=self.grid.dtype)
            '''
            for i, cols in enumerate(zip(self.grid.T, grid_input.T)):
                (i1, ) = cupy.nonzero(cols[0])
                (i2, ) = cupy.nonzero(cols[1])
                #if cupy.all(self.in1d(i2,i1)):
                if cupy.all(self.impl(cols[0],cols[1])):
                    # TODO: finish the reduce operation
                    #if i1.size == i2.size:
                    pass
                        # grid_output[0:255, i] =
                else:
                    raise AssociativeMemoryError('Applicability '
                                                 'condition error.')
              '''
            grid_imp=self.impl(grid_input,self.grid)
            if cupy.all(grid_imp):
                return True
            else:
                #raise AssociativeMemoryError('Applicability condition error.')
                return False

                    
            
            #return grid_input
