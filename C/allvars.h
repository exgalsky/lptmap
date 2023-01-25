#define NUMMAPCODES 5

#define KAPCODE 0
#define KSZCODE 1
#define TAUCODE 2
#define DTBCODE 3
#define CIBCODE 4

#ifndef ALLVARS_H
#define ALLVARS_H

#include <mpi.h>
#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>

#ifdef NOTYPEPREFIX_FFTW
#include        <rfftw_mpi.h>
#else
#ifdef DOUBLEPRECISION_FFTW
#include     <drfftw_mpi.h>	/* double precision FFTW */
#else
#include     <srfftw_mpi.h>
#endif
#endif

// IO
extern FILE *logfile;

// MPI Variables
extern int myid, nproc;

// FFTW Variables
extern rfftwnd_mpi_plan plan, iplan; 
extern int local_nx, local_x_start, 
           local_ny_after_transpose, local_y_start_after_transpose, 
           total_local_size;

// Slab sizes
extern long Nlocal;             // Local slab dimension 
extern long int size;           // Local slab size 
extern long int size_fftw;

// Arrays
extern fftw_real *delta, *delta1, *delta2, *sx1, *sy1, *sz1, *sx2, *sy2, *sz2;


// Parameters
extern struct Parameter{
  float BoxSize;
  int verbose, N, lptcode;
  char DeltaFile[256], BaseOut[256];
} Parameters;

// Derived parameters
extern float ovrt, oram;

#endif

