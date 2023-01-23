#include <stdio.h>
#include "allvars.h"

// IO
FILE *logfile;

// MPI Variables
int myid, nproc;

// FFTW Variables
rfftwnd_mpi_plan plan, iplan; 
int local_nx, local_x_start, 
           local_ny_after_transpose, local_y_start_after_transpose, 
           total_local_size;

// Slab sizes
long Nlocal;    // Local slab dimension 
long int size;      // Local slab size 
long int size_fftw;

// Arrays
fftw_real *delta, *delta1, *delta2, *sx1, *sy1, *sz1, *sx2, *sy2, *sz2;

// Parameters 
struct Parameter Parameters;

// Derived parameters
float ovrt, oram;
