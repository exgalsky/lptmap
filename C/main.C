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
  ReadParameterFile();

  // Set default parameters
  SetDefaultParameters();
  
  int N=Parameters.N;

  DInit=growth(Parameters.zInit, Parameters.Omegam, Parameters.Omegal, Parameters.w);

  if(Parameters.ReadDisp==0){
    // Make FFTW plans
    plan  = rfftw3d_mpi_create_plan(MPI_COMM_WORLD,N, N, N,
				  FFTW_REAL_TO_COMPLEX, FFTW_ESTIMATE);
    iplan = rfftw3d_mpi_create_plan(MPI_COMM_WORLD,N, N, N,
				  FFTW_COMPLEX_TO_REAL, FFTW_ESTIMATE);
  
    rfftwnd_mpi_local_sizes(plan, &local_nx, &local_x_start,
			  &local_ny_after_transpose,
			  &local_y_start_after_transpose,
			  &total_local_size);
  } else {
    total_local_size = N*N*(N/nproc);
  }

  ovrt=0;
  oram=0;
  GetOverhead(total_local_size,&ovrt,&oram);
  ReportMemory("after plan creation",total_local_size,ovrt,oram);

  // Allocate arrays
  AllocateArrays();
  ReportMemory("after array allocation",size_fftw,ovrt,oram);

  if(Parameters.ReadDisp==0){
    // Read linear density contrast
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

    // Write LPT displacements
    WriteDisplacements();
    MPI_Barrier(MPI_COMM_WORLD);

  } else {

    // Read LPT displacements 
    char filename[256];
    if(Parameters.lptcode>0){
      sprintf(filename,"%s/sx1_%s",Parameters.DispPath,Parameters.DispSuffix); ReadGridFromFile(sx1, filename);
      sprintf(filename,"%s/sy1_%s",Parameters.DispPath,Parameters.DispSuffix); ReadGridFromFile(sy1, filename);
      sprintf(filename,"%s/sz1_%s",Parameters.DispPath,Parameters.DispSuffix); ReadGridFromFile(sz1, filename);
    }
    if(Parameters.lptcode>1){
      sprintf(filename,"%s/sx2_%s",Parameters.DispPath,Parameters.DispSuffix); ReadGridFromFile(sx2, filename);
      sprintf(filename,"%s/sy2_%s",Parameters.DispPath,Parameters.DispSuffix); ReadGridFromFile(sy2, filename);
      sprintf(filename,"%s/sz2_%s",Parameters.DispPath,Parameters.DispSuffix); ReadGridFromFile(sz2, filename);
    }

  }
  
  if(Parameters.MapCode>0){
    // Make maps
    MakeMaps();
    MPI_Barrier(MPI_COMM_WORLD);
        
    // Write maps
    WriteMaps();
    MPI_Barrier(MPI_COMM_WORLD);    
  }
   
  // Finalize and return
  MPI_Finalize();  if(myid==0) printf("\n\n"); 
  
  fclose(stdout);
  exit(0);

}

