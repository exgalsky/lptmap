#include <math.h>
#include "proto.h"
#include "allvars.h"
#include <unistd.h>

void SetDefaultParameters(){
  Parameters.Periodicity = Parameters.BoxSize;
}

void usage(){

  if(myid==0){
    printf("\n usage: websky-lpt [options]\n");
    
    printf("\n OPTIONS:\n");
    printf("\n   -h show this message    [default = OFF]");
    printf("\n   -v verbose              [default = OFF]");
    printf("\n   -r read displacements   [default = OFF]");
    printf("\n   -N grid dimension       [default = 512]");
    printf("\n   -b box size in Mpc      [default = 100 Mpc]");
    printf("\n   -l LPT code, [l]LPT     [default = 1 --> 1LPT");
    printf("\n   -m binary map code      [default = 0; e.g. kap=0 --> no kappa ; kap=1 --> do kappa]");
    printf("\n                            = kap * 1  (CMB lensing)");
    printf("\n                            + ksz * 2  (kinetic SZ) ");
    printf("\n                            + tau * 4  (tau_es)     ");
    printf("\n   -d density file         [default = './delta']");
    printf("\n   -P displacements path   [default = './']");
    printf("\n   -s displacements suffix [default = '7700Mpc_n6144_nb30_nt16_no768']");
    printf("\n   -M map suffix           [default = '']");
    printf("\n   -p parameter file       [default = './lptmap.param']");
    printf("\n\n");
  }

  MPI_Finalize();
  exit(0);

}

void CommandLine(int argc, char *argv[])
{

  int c;

  opterr=0;
  
  Parameters.Verbose      = 0;
  Parameters.ReadDisp     = 0;
  Parameters.BoxSize      = 100;
  Parameters.N            = 512;
  Parameters.lptcode      = 1;
  Parameters.MapCode      = 0;

  sprintf(Parameters.DeltaFile,"./delta");
  sprintf(Parameters.DispPath,"./");
  sprintf(Parameters.DispSuffix,"7700Mpc_n6144_nb30_nt16_no768");
  sprintf(Parameters.MapSuffix,"");
  sprintf(Parameters.ParamFile,"./lptmap.param");

  int i=0;
  while ((c = getopt (argc, argv, "hvrN:b:l:m:d:M:P:s:p:")) != -1){
    switch (c)
      {
      case 'h':
	      usage();
	      break;
      case 'v':
	      Parameters.Verbose = 1;
	      break;
      case 'r':
  	    Parameters.ReadDisp = 1;
  	    break;
      case 'N':
	      Parameters.N = atoi(optarg);
	      break;
      case 'b':
	      Parameters.BoxSize = atof(optarg);
	      break;
      case 'l':
	      Parameters.lptcode = atoi(optarg);
	      break;
      case 'm':
    	  Parameters.MapCode = atoi(optarg);
    	  break;
      case 'd':
  	    sprintf(Parameters.DeltaFile,"%s",optarg);
  	    break;
      case 'P':
    	  sprintf(Parameters.DispPath,"%s",optarg);
    	  break;
      case 's':
  	    sprintf(Parameters.DispSuffix,"%s",optarg);
  	    break;
      case 'M':
    	  sprintf(Parameters.MapSuffix,"%s",optarg);
    	  break;
      case 'p':
    	  sprintf(Parameters.ParamFile,"%s",optarg);
    	  break;
      default:
	      usage();
      }
    }
  
  if(Parameters.Verbose == 0){
    // Redirect stdout to output file
    char fname[256];
    sprintf(fname,"lptmap-C.stdout");
    freopen(fname,"w",stdout);
  }
  FillMapCodeArray(2*Parameters.MapCode,NUMMAPCODES);
  int *buff = new int[NUMMAPCODES];
  for (int j=0;j<NUMMAPCODES;j++) buff[j]=Parameters.DoMap[j];
  for (int j=0;j<NUMMAPCODES;j++) Parameters.DoMap[j]=buff[NUMMAPCODES-j-1];
  
  // Don't buffer stdout and stderr
  setvbuf(stdout, NULL, _IONBF, 0);
  setvbuf(stderr, NULL, _IONBF, 0);

  if(myid==0){
    printf("\n Command line:");
    for(int i=0;i<argc;i++) printf(" %s",argv[i]); printf("\n");  
  }

  return;

}

void FillMapCodeArray(int n, int m)
{
  if (n < 0 || n > pow(2,NUMMAPCODES+1)){
    if(myid==0) fprintf (stderr, "\n map code %d not an allowed value\n",n);
    MPI_Barrier(MPI_COMM_WORLD);
    usage();
  }
  if (n / 2 != 0) {
    FillMapCodeArray(n / 2,m-1);
  }
  Parameters.DoMap[m] = n % 2;
}