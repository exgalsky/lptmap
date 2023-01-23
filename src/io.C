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
  
  if(myid==0) printf("\n writing displacements\n",myid,Nlocal);

  int N=Parameters.N;
  float BoxSize=Parameters.BoxSize;

  float *slab = new float[N*N];

  long blocksize = (long)N * (long)N * (long)N * (long)(sizeof(float));
  long offset_inblock = (long)myid*(long)N*(long)N*(long)Nlocal*(long)(sizeof(float));
  long bsize = N*N*sizeof(float);
  long offset;

  char fnamelpt[256];

  if (Parameters.lptcode==1){
    sprintf(fnamelpt,"%s_1lpt.bin",Parameters.BaseOut);
  } else {
    sprintf(fnamelpt,"%s_2lpt.bin",Parameters.BaseOut);    
  }

  offset = 0*blocksize + offset_inblock;
  for(long i=0;i<Nlocal;i++){for(long j=0;j<N;j++){for(long k=0;k<N;k++){
    long index = i*N*(N+2)+j*(N+2)+k; slab[j*N+k] = sx1[index];}
  }
  parallel_write(fnamelpt,bsize, offset, slab);offset+=bsize;}

  offset = 1*blocksize + offset_inblock;
  for(long i=0;i<Nlocal;i++){for(long j=0;j<N;j++){for(long k=0;k<N;k++){
    long index = i*N*(N+2)+j*(N+2)+k; slab[j*N+k] = sy1[index];}
  }
  parallel_write(fnamelpt,bsize, offset, slab);offset+=bsize;}

  offset = 2*blocksize + offset_inblock;
  for(long i=0;i<Nlocal;i++){for(long j=0;j<N;j++){for(long k=0;k<N;k++){
    long index = i*N*(N+2)+j*(N+2)+k; slab[j*N+k] = sz1[index];}
  }
  parallel_write(fnamelpt,bsize, offset, slab);offset+=bsize;}

  if (Parameters.lptcode==2){
    offset = 3*blocksize + offset_inblock;
    for(long i=0;i<Nlocal;i++){for(long j=0;j<N;j++){for(long k=0;k<N;k++){
      long index = i*N*(N+2)+j*(N+2)+k; slab[j*N+k] = sx2[index];}
    }
    parallel_write(fnamelpt,bsize, offset, slab);offset+=bsize;}

    offset = 4*blocksize + offset_inblock;
    for(long i=0;i<Nlocal;i++){for(long j=0;j<N;j++){for(long k=0;k<N;k++){
      long index = i*N*(N+2)+j*(N+2)+k; slab[j*N+k] = sy2[index];}
    }
    parallel_write(fnamelpt,bsize, offset, slab);offset+=bsize;}

    offset = 5*blocksize + offset_inblock;
    for(long i=0;i<Nlocal;i++){for(long j=0;j<N;j++){for(long k=0;k<N;k++){
      long index = i*N*(N+2)+j*(N+2)+k; slab[j*N+k] = sz2[index];}
    }
    parallel_write(fnamelpt,bsize, offset, slab);offset+=bsize;}
  }
  delete[] slab;

  if(myid==0) printf("\n LPT data written...");
}
