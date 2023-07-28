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
#include "chealpix.h"

#define FLOAT 1
#define DOUBLE 2
#define STRING 3
#define INT 4
#define MAXTAGS 300

void WriteSingleMap(float *, char *);

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

  if(myid==0) printf("\n grid data read from %s",fname);

}

void WriteDisplacements()
{
  
  int N=Parameters.N;
  float BoxSize=Parameters.BoxSize;

  long offset = (long)N*(long)N*(long)Nlocal*(long)(sizeof(float))*(long)myid;
  long bsize =  (long)N*(long)N*(long)Nlocal*(long)(sizeof(float));
  int doffset;

  char fsx1[256], fsy1[256], fsz1[256];
  char fsx2[256], fsy2[256], fsz2[256];

  float *slab = delta1;
  if(myid==0) printf("\n writing LPT data...");

  sprintf(fsx1,"%s/sx1_%s",Parameters.DispPath,Parameters.DispSuffix);
  sprintf(fsy1,"%s/sy1_%s",Parameters.DispPath,Parameters.DispSuffix);
  sprintf(fsz1,"%s/sz1_%s",Parameters.DispPath,Parameters.DispSuffix);

  if (Parameters.lptcode==2){
    sprintf(fsx2,"%s/sx2_%s",Parameters.DispPath,Parameters.DispSuffix);
    sprintf(fsy2,"%s/sy2_%s",Parameters.DispPath,Parameters.DispSuffix);
    sprintf(fsz2,"%s/sz2_%s",Parameters.DispPath,Parameters.DispSuffix);
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

void ReadParameterFile()
{

  FILE *fd, *fdout;
  char buf[200], buf1[200], buf2[200], buf3[400];
  char fname[256];
  int i, j, nt;
  int id[MAXTAGS];
  void *addr[MAXTAGS];
  char tag[MAXTAGS][50];

  sprintf(fname,Parameters.ParamFile);

  nt=0;

  strcpy(tag[nt], "Omegam");
  addr[nt] = &Parameters.Omegam;
  id[nt++] = FLOAT;

  strcpy(tag[nt], "Omegab");
  addr[nt] = &Parameters.Omegab;
  id[nt++] = FLOAT;

  strcpy(tag[nt], "Omegal");
  addr[nt] = &Parameters.Omegal;
  id[nt++] = FLOAT;

  strcpy(tag[nt], "h");
  addr[nt] = &Parameters.h;
  id[nt++] = FLOAT;

  strcpy(tag[nt], "w");
  addr[nt] = &Parameters.w;
  id[nt++] = FLOAT;

  strcpy(tag[nt], "zKappa");
  addr[nt] = &Parameters.zKappa;
  id[nt++] = FLOAT;

  strcpy(tag[nt], "zInit");
  addr[nt] = &Parameters.zInit;
  id[nt++] = FLOAT;

  strcpy(tag[nt], "InitialRedshift");
  addr[nt] = &Parameters.InitialRedshift;
  id[nt++] = FLOAT;

  strcpy(tag[nt], "FinalRedshift");
  addr[nt] = &Parameters.FinalRedshift;
  id[nt++] = FLOAT;

  strcpy(tag[nt], "Nside");
  addr[nt] = &Parameters.Nside;
  id[nt++] = INT;

  if((fd = fopen(fname, "r")))
    {
      sprintf(buf, "%s%s", fname, "-usedvalues");
      if(!(fdout = fopen(buf, "w")))
        {
          if(myid==0) printf("error opening file '%s' \n", buf);
        }
      else
        {
          if(myid>0) fclose(fdout);
          while(!feof(fd))
            {
              *buf = 0;
              fgets(buf, 200, fd);
              if(sscanf(buf, "%s%s%s", buf1, buf2, buf3) < 2)
                continue;

              if(buf1[0] == '%' || buf1[0] == '#')
                continue;

              for(i = 0, j = -1; i < nt; i++)
                if(strcmp(buf1, tag[i]) == 0)
                  {
                    j = i;
                    tag[i][0] = 0;
                    break;
                  }

              if(j >= 0)
                {
                  switch (id[j])
                    {
                    case FLOAT:
                      *((float *) addr[j]) = atof(buf2);
                      if(myid==0) fprintf(fdout, "%-35s%f\n", buf1, *((float *) addr[j]));
                      break;
                    case DOUBLE:
                      *((float *) addr[j]) = atof(buf2);
                      if(myid==0) fprintf(fdout, "%-35s%f\n", buf1, *((float *) addr[j]));
                      break;
                    case STRING:
                      strcpy((char *)addr[j], buf2);
                      if(myid==0) fprintf(fdout, "%-35s%s\n", buf1, buf2);
                      break;
                    case INT:
                      *((int *) addr[j]) = atoi(buf2);
                      if(myid==0) fprintf(fdout, "%-35s%d\n", buf1, *((int *) addr[j]));
                      break;
                    }
                }
              else
                {
                  fprintf(stdout,
                          "Error in %s: Tag '%s' not allowed or multiple defined.\n",
                          fname, buf1);
                }
            }
          fclose(fd);
          if(myid==0) fclose(fdout);

        }
    }
  else
    {
      printf("\nParameter file %s not found.\n\n", fname);
    }

   if(myid==0) printf("\n Parameter file read...");

#undef DOUBLE
#undef STRING
#undef INT
#undef MAXTAGS

}

void WriteMaps()
{

  if(myid==0){

    printf("\n Writing maps...\n");

    if(Parameters.DoMap[KAPCODE]) WriteSingleMap( kapmap,"kap");
    if(Parameters.DoMap[KSZCODE]) WriteSingleMap (kszmap,"ksz");
    if(Parameters.DoMap[TAUCODE]) WriteSingleMap (taumap,"tau");

    printf("\n Maps written...");

  }

  MPI_Barrier(MPI_COMM_WORLD);
  return;

}

void WriteSingleMap(float *map, char *base){

  char coord[1];
  FILE *fout;
  int  mapsize = Parameters.Nside*Parameters.Nside*12;

  // binary format
  char binary_fname[256];
  sprintf(binary_fname,"%s_%s.bin",Parameters.MapSuffix,base);
  fout = fopen(binary_fname,"wb");
  fwrite(map,4,mapsize,fout);
  fclose(fout);

  // fits format
  char fits_fname[256];
  sprintf(fits_fname,"!%s_%s.fits",Parameters.MapSuffix,base);
  sprintf(coord,"C");
  write_healpix_map(map, Parameters.Nside, fits_fname, 1, coord);

}
