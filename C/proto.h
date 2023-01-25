#ifndef PROTO_H
#define PROTO_H

#include <stdio.h>
#include "memorytracking.h"
#include "parallel_io.h"

#ifndef ALLVARS_H
#include "allvars.h"
#endif

// arrayoperations.C
void AllocateArrays(); 

// commandline.C
void usage();
void CommandLine(int, char **);

// lpt.C
void Displace_1LPT(float *, float *, float *, float *);
void Displace_2LPT(float *, float *, float *, float *, float *, float *, float *, float *);

// io.C
void ReadGridFromFile(float *, char *);
void WriteDisplacements();

#endif

