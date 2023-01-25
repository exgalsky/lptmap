#include "allvars.h"
#include "proto.h"
#include <math.h>

// Output 1st [and 2nd] order LPT displacements given input density contrast

int main(int argc, char *argv[])
{
  
  // Initialize MPI
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  // Parse command line  
  CommandLine(argc, &argv[0]);

  // Read parameter file
  int N=Parameters.N;

  // Make FFTW plans
  plan  = rfftw3d_mpi_create_plan(MPI_COMM_WORLD,N, N, N,
				  FFTW_REAL_TO_COMPLEX, FFTW_ESTIMATE);
  iplan = rfftw3d_mpi_create_plan(MPI_COMM_WORLD,N, N, N,
				  FFTW_COMPLEX_TO_REAL, FFTW_ESTIMATE);
  
  rfftwnd_mpi_local_sizes(plan, &local_nx, &local_x_start,
			  &local_ny_after_transpose,
			  &local_y_start_after_transpose,
			  &total_local_size);

  ovrt=0;
  oram=0;
  GetOverhead(total_local_size,&ovrt,&oram);
  ReportMemory("after plan creation",total_local_size,ovrt,oram);

  // Allocate arrays
  AllocateArrays();
  ReportMemory("after array allocation",size_fftw,ovrt,oram);

  // Read density contrast
  ReadGridFromFile( delta1, Parameters.DeltaFile);
  MPI_Barrier(MPI_COMM_WORLD);

  // Calculate LPT displacements
  if(Parameters.lptcode>1){
    Displace_2LPT(delta1, delta2, sx1, sy1, sz1, sx2, sy2, sz2);
    Displace_1LPT(delta2, sx2, sy2, sz2);
    ReadGridFromFile( delta1, Parameters.DeltaFile);
  }
  if(Parameters.lptcode>0) Displace_1LPT(delta1, sx1, sy1, sz1);  
  MPI_Barrier(MPI_COMM_WORLD);

  // Write displacements
  WriteDisplacements();
  MPI_Barrier(MPI_COMM_WORLD);

  // Finalize and return
  MPI_Finalize();  if(myid==0) printf("\n\n"); 
  
  fclose(stdout);
  exit(0);

}

