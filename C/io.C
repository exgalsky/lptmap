#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include "proto.h"
#include "allvars.h"

void WriteDisplacements(float *, char *);

void ReadGridFromFile(float *grid, char *fname)
{
    
  int N=Parameters.N;

  float *slab = new float[N*N];

  long offset = (long)myid*(long)N*(long)N*(long)Nlocal*(long)sizeof(float);
  long bsize = N*N*sizeof(float);

  FILE *fd = fopen(fname,"rb");
  if(fd==NULL){
    printf("could not open grid file %s\n",fname);
    exit(0);
    MPI_Finalize();
  }

  for(long i=0;i<Nlocal;i++){
    // Read this slab from the input file

    long int offset_local = i*N*N*sizeof(float) + offset;
    parallel_read(fname, bsize, offset_local, slab);

    for(long j=0;j<N;j++){
      for(long k=0;k<N;k++){
        long index = i*N*(N+2)+j*(N+2)+k;
        grid[index] = slab[j*N+k]; 
      }
    }

  }

  delete[] slab;

  if(myid==0) printf("\n grid data read...");

}

void WriteDisplacements()
{
  
  int N=Parameters.N;
  float BoxSize=Parameters.BoxSize;

  long offset = (long)N*(long)N*(long)Nlocal*(long)(sizeof(float))*(long)myid;
  long bsize =  (long)N*(long)N*(long)Nlocal*(long)(sizeof(float))
  int doffset;

  char fsx1[256], fsy1[256], fsz1[256];
  char fsx2[256], fsy2[256], fsz2[256];

  float *slab = delta1;
  if(myid==0) printf("\n writing LPT data...");

  sprintf(fsx1,"./sx1_%s",Parameters.BaseOut);
  sprintf(fsy1,"./sy1_%s",Parameters.BaseOut);
  sprintf(fsz1,"./sz1_%s",Parameters.BaseOut);
  if (Parameters.lptcode==2){
    sprintf(fsx2,"./sx2_%s",Parameters.BaseOut);
    sprintf(fsy2,"./sy2_%s",Parameters.BaseOut);
    sprintf(fsz2,"./sz2_%s",Parameters.BaseOut);
  }

  // 1lpt x
  if(myid==0) printf("\n writing sx1");
  for(long i=0;i<Nlocal;i++){
    for(long j=0;j<N;j++){for(long k=0;k<N;k++){long index = i*N*(N+2)+j*(N+2)+k; slab[i*N*N+j*N+k] = sx1[index];}}
  }
  parallel_write(fsx1, bsize, offset, slab);

  // 1lpt y
  if(myid==0) printf("\n writing sy1");
  for(long i=0;i<Nlocal;i++){
    for(long j=0;j<N;j++){for(long k=0;k<N;k++){long index = i*N*(N+2)+j*(N+2)+k; slab[i*N*N+j*N+k] = sy1[index];}}
  }
  parallel_write(fsy1, bsize, offset, slab);
  // 1lpt z
  if(myid==0) printf("\n writing sz1");
  for(long i=0;i<Nlocal;i++){
    for(long j=0;j<N;j++){for(long k=0;k<N;k++){long index = i*N*(N+2)+j*(N+2)+k; slab[i*N*N+j*N+k] = sz1[index];}}
  }
  parallel_write(fsz1, bsize, offset, slab);

  if (Parameters.lptcode==2){
    // 2lpt x
    if(myid==0) printf("\n writing sx2");
    for(long i=0;i<Nlocal;i++){
      for(long j=0;j<N;j++){for(long k=0;k<N;k++){long index = i*N*(N+2)+j*(N+2)+k; slab[i*N*N+j*N+k] = sx2[index];}}
    }
    parallel_write(fsx2, bsize, offset, slab);
    // 2lpt y
    if(myid==0) printf("\n writing sy2");
    for(long i=0;i<Nlocal;i++){
      for(long j=0;j<N;j++){for(long k=0;k<N;k++){long index = i*N*(N+2)+j*(N+2)+k; slab[i*N*N+j*N+k] = sy2[index];}}
    }
    parallel_write(fsy2, bsize, offset, slab);
    // 2lpt z
    if(myid==0) printf("\n writing sz2");
    for(long i=0;i<Nlocal;i++){
      for(long j=0;j<N;j++){for(long k=0;k<N;k++){long index = i*N*(N+2)+j*(N+2)+k; slab[i*N*N+j*N+k] = sz2[index];}}
    }
    parallel_write(fsz2, bsize, offset, slab);
  }
  delete[] slab;

  if(myid==0) printf("\n ...LPT data written");
}
