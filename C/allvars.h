#define NUMMAPCODES 4

#define KAPCODE 0
#define KSZCODE 1
#define TAUCODE 2
#define DTBCODE 3

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

// Map size
extern long mapsize;

// Arrays
extern fftw_real *delta, *delta1, *delta2, *sx1, *sy1, *sz1, *sx2, *sy2, *sz2;
extern float *taumap,  *kszmap,  *kapmap;
extern float *taumapl, *kszmapl, *kapmapl;

// Parameters
extern struct Parameter{
  float Omegam, Omegab, Omegal, h, w, zInit, zKappa;
  float BoxSize, Periodicity, InitialRedshift, FinalRedshift;
  int Verbose, ReadDisp, N, Nside, lptcode, Evolve;
  char DeltaFile[256], DispPath[256], DispSuffix[256], MapSuffix[256], ParamFile[256];
  int DoMap[NUMMAPCODES], CurrentCode, MapCode;
} Parameters;

// Derived parameters
extern float DInit, ovrt, oram;

#endif

