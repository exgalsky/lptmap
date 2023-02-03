import numpy as np
from mpi4py import MPI
from webskylpt.io import read_displacements
from webskylpt.map import makemap

def main(args=None, comm=None):
    comm = MPI.COMM_WORLD
    rank = params.comm.Get_rank()
    size = params.comm.Get_size()

    s = read_displacement_coefficients(args.input,comm=comm,rank=rank)
    
    map = makemap(s,comm=comm,rank=rank)