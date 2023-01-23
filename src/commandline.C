#include <math.h>
#include "proto.h"
#include "allvars.h"
#include <unistd.h>

void usage(){

  if(myid==0){
    printf("\n usage: websky-lpt [options]\n");
    
    printf("\n OPTIONS:\n");
    printf("\n   -h show this message");
    printf("\n   -v verbose            [default = OFF]"); 
    printf("\n   -D density file       [default = './delta'");
    printf("\n   -o Output base name   [default = './s']");
    printf("\n   -N grid dimension     [default = 512]");
    printf("\n   -B box size in Mpc    [default = 100 Mpc]");
    printf("\n   -l LPT code, [l]LPT   [default = 1 --> 1LPT");
    printf("\n\n");
  }

  MPI_Finalize();
  exit(0);

}

void CommandLine(int argc, char *argv[])
{

  int c;

  opterr=0;
  
  sprintf(Parameters.DeltaFile,"./delta");
  sprintf(Parameters.BaseOut,"./s");
  
  Parameters.verbose      = 0;
  Parameters.BoxSize      = 100;
  Parameters.N            = 512;
  Parameters.lptcode      = 1; 
  
  while ((c = getopt (argc, argv, "hvD:o:N:B:l:")) != -1)
    switch (c)
      {
      case 'h':
	      usage();
	      break;
      case 'v':
	      Parameters.verbose = 1;
	      break;
      case 'D':
	      sprintf(Parameters.DeltaFile,"%s",optarg);
	      break;
      case 'o':
	      sprintf(Parameters.BaseOut,"%s",optarg);
	      break;
      case 'N':
	      Parameters.N = atoi(optarg);
	      break;
      case 'B':
	      Parameters.BoxSize = atof(optarg);
	      break;
      case 'l':
	      Parameters.lptcode = atoi(optarg);
	      break;
      case '?':
	      if (optopt == 'i'){
	         if(myid==0) fprintf (stderr, "\n Option -%c requires an argument.\n", optopt);
	         usage();
	      }
	      if (optopt == 'o'){
	         if(myid==0) fprintf (stderr, "\n Option -%c requires an argument.\n", optopt);
	         usage();
	      }
	      else if (isprint (optopt)){
	         if(myid==0) fprintf (stderr, "\n Unknown option `-%c'.\n", optopt);
	         usage();
	      }
	      else{
	         if(myid==0) fprintf (stderr,
		           "Unknown option character `\\x%x'.\n",optopt);
	         usage();
	      }
	      return;
      default:
	      usage();
      }

  if(Parameters.verbose == 0){
    // Redirect stdout to output file
    char fname[256];
    sprintf(fname,"%s.stdout",Parameters.BaseOut);
    freopen(fname,"w",stdout);
  }

  // Don't buffer stdout and stderr
  setvbuf(stdout, NULL, _IONBF, 0);
  setvbuf(stderr, NULL, _IONBF, 0);

  if(myid==0){
    printf("\n Command line:");
    for(int i=0;i<argc;i++) printf(" %s",argv[i]); printf("\n");  
  }

  Parameters.N = Parameters.N;
  Parameters.BoxSize = Parameters.BoxSize;
  return;

}
