import mpi_utils as mutl
import jax_utils as jutl 
import numpy as np

class backend:
    def __init__(self, force_no_mpi=False, force_no_gpu=False):
        self.mpi_backend = mutl.mpi_handler(force_no_mpi=force_no_mpi)
        self.jax_backend = jutl.jax_handler(force_no_gpu=force_no_gpu)

    def datastream_setup(self, data_shape, bytes_per_cell, peak_per_cell_memory, jax_overhead_factor, decom_type='slab', divide_axis=0):
        self.mpi_start, self.mpi_stop = self.mpi_backend.divide4mpi(data_shape, decom_type=decom_type, divide_axis=divide_axis)
        self.mpi_offset = self.mpi_backend.data_offset(data_shape, bytes_per_cell, divide_axis=divide_axis, decom_type=decom_type)
        
        self.chunk_shape = list(data_shape).copy()
        self.chunk_shape[divide_axis] = self.mpi_backend.slab_per_Proc[self.mpi_backend.id]
        self.chunk_shape = tuple(self.chunk_shape)

        self.jax_backend.jax_tasks(self.chunk_shape,peak_per_cell_memory, jax_overhead_factor, divide_axis=divide_axis)
        self.jax_backend.jax_data_offset(self.chunk_shape,bytes_per_cell, mpi_offset=self.mpi_offset, divide_axis=divide_axis,decom_type=decom_type)

        self.__jslice_shape = list(self.chunk_shape)
        self.__jslice_shape[divide_axis] = -1

    def get_iterator(self):

        iterator = []
        for ijax in range(self.jax_backend.n_jaxcalls):
            start = self.mpi_start + np.sum(self.jax_backend.slices_per_jaxcall[0:ijax])
            stop = start + self.jax_backend.slices_per_jaxcall[ijax]

            offset = self.jax_backend.offsets_per_call[ijax]

            shape = self.__jslice_shape.copy()
            shape[shape == -1] = self.jax_backend.slices_per_jaxcall[ijax]
            shape = tuple(shape)

            iterator.append([start, stop, offset, shape])

        return iterator
        




