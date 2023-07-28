#ifndef PROTO_H
#define PROTO_H

#include <stdio.h>
#include "cosmology.h"
#include "memorytracking.h"
#include "parallel_io.h"

#ifndef ALLVARS_H
#include "allvars.h"
#endif

// arrayoperations.C
void AllocateArrays(); 

// parameters.C
void usage();
void CommandLine(int, char **);
void FillMapCodeArray(int, int);

// lpt.C
void Displace_1LPT(float *, float *, float *, float *);
void Displace_2LPT(float *, float *, float *, float *, float *, float *, float *, float *);

// io.C
void ReadGridFromFile(float *, char *);
void WriteDisplacements();
void ReadParameterFile();
void SetDefaultParameters();
void WriteMaps();

// makemaps.C
void MakeMaps();

#endif

