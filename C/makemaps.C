#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "proto.h"
#include "allvars.h"
#include "tables.h"
#include "globaltablevars.h"
#include "chealpix.h"

void  ReportMapStatistics(float *, int, char *);
int   SlabShellOverlap(float **, float, float);

/* ************* ************* ************* ************* ************* ************* */
/* ************* ************* ************* ************* ************* ************* */

void MakeMaps()
{
  
  MPI_Barrier(MPI_COMM_WORLD);

  double t1 = MPI_Wtime();
    
  int i,j,k,index;
  int i1,j1,k1,i2,j2,k2;
  
  float xm,ym,zm;    // position in map coordinates
  float x,y,z;       // position in cube coordinates
  
  double theta,phi;   // angle in cube coordinates
  float deg2rad=2.*M_PI/360.;

  float Omegam = Parameters.Omegam;
  float Omegab = Parameters.Omegab;
  float Omegal = 1-Omegam;
  float      h = Parameters.h;
  float  zInit = Parameters.zInit;
  float zKappa = Parameters.zKappa;

  float Tcmb = 2.726e6; // Tcmb in muK
  float hoc = h * 1e2 / 3e5; // H0/c in units of 1/Mpc

  float thompson = 6.65e-25;
  float YHe = 0.25;
  float hubble0 = 100./3.086e19*h;
  float ne0=Omegab*h*h*1.88e-29/1.67e-24;
  float c=3e10;

  float dtau0=thompson*ne0*c/hubble0;

  float NHe;

  // Set Radius to Redshift Table
  
  Radius2RedshiftTable = new double[NRTABLE];
  SetRadius2RedshiftTable(h, Omegam, Omegal, Radius2RedshiftTable);

  // Set Redshift to Radius Table
  
  Redshift2RadiusTable = new double[NZTABLE];
  SetRedshift2RadiusTable(h, Omegam, Omegal, Redshift2RadiusTable);

  // Set Redshift to WKappa Table
  
  Redshift2WKappaTable = new double[NZTABLE];
  SetRedshift2WKappaTable(h, Omegam, Omegal, zKappa, Redshift2WKappaTable, 
			  Redshift2RadiusTable);

  // Set Redshift to WTau Table
  
  Redshift2WTauTable = new double[NZTABLE];
  SetRedshift2WTauTable(h, Omegam, Omegal, Redshift2WTauTable);

  // Local copies of parameters

  float BoxSize     = Parameters.BoxSize;
  float Periodicity = Parameters.Periodicity;

  float x0,y0,z0; // position of corner of box so it spans [x0,y0,z0] to [x0+boxsize,y0+boxsize,z0+boxsize], in Mpc
  float xc0,yc0,zc0; // position of first box center
  float xc1,yc1,zc1,xc2,yc2,zc2;

  // set shell and periodicity
  float zmin = Parameters.InitialRedshift;
  float zmax = Parameters.FinalRedshift;
  if(myid==0) printf("zmin=%f zmax=%f\n",zmin,zmax);

  float rmin = Redshift2Float(zmin,Redshift2RadiusTable);
  float rmax = Redshift2Float(zmax,Redshift2RadiusTable);
  int nperiodic = (int)(2*rmax / BoxSize + 1);
  int tiledbox = nperiodic * BoxSize ; 
  if(myid==0) printf("rmin = %f rmax = %f nperiodic = %d BoxSize = %f\n",rmin,rmax,nperiodic,BoxSize);

  x0  = - tiledbox / 2     ; y0  = - tiledbox / 2     ; z0  = - tiledbox / 2     ;
  xc0 = x0 + BoxSize / 2   ; yc0 = y0 + BoxSize / 2   ; zc0 = z0 + BoxSize / 2 ; 
  
  if(myid==0) printf("x0,y0,z0 = %f,%f,%f\n",x0,y0,z0);

  int N=Parameters.N;

  // if (nperiodic <= 2) rmax = fminf(rmax,BoxSize);
  
  int Nxmin = Nlocal * myid;
  float slabsize = BoxSize / nproc;
  float slab_xoffset = slabsize * myid ;

  // local copies of maps

  MPI_Barrier(MPI_COMM_WORLD);

  float *kapmapl, *kszmapl, *taumapl, *dtbmapl;
  //Local copies of maps
  if(Parameters.DoMap[KAPCODE]==1) kapmapl = new float[mapsize](); 
  if(Parameters.DoMap[KSZCODE]==1) kszmapl = new float[mapsize]();
  if(Parameters.DoMap[TAUCODE]==1) taumapl = new float[mapsize]();

  ReportMemory("before map projection",total_local_size,ovrt,oram);

  if(myid==0){
    float dztable = ((float)ZTABLE_FINAL - ZTABLE_INITIAL) / NZTABLE;
    FILE* wfile = fopen("weights.dat","w");
    for(int i=0; i<NZTABLE; i++){
      float zcur  = ZTABLE_INITIAL + i*dztable;

      float wkapt = Redshift2WKappaTable[i];
      float wkap  = Redshift2Float(zcur,Redshift2WKappaTable);

      float wtaut = Redshift2WTauTable[i];
      float wtau  = Redshift2Float(zcur,Redshift2WTauTable);
      
      float dcur  = growth(zcur,Parameters.Omegam,Parameters.Omegal, Parameters.w)/DInit;

      fprintf(wfile,"%e %e %e %e %e %e\n",zcur,wkap,wkapt,wtau,wtaut,dcur);
    }
    fclose(wfile);
  }

  float CellSize = BoxSize / N ;
  float CellVolume = CellSize*CellSize*CellSize;

  // before looping over periodic slabs, find out maximum number of 
  // images in each dimension use 15 Gpc as largest possible radius

  double *vec = new double[3];
  double corner[3];
  float **bb = new float*[2];
  bb[0] = new float[3];
  bb[1] = new float[3];

  // slab corners

  float xs1,ys1,zs1,xs2,ys2,zs2;

  int iLPT = Parameters.lptcode;

  // Each process loops over all periodic slab images overlapping with 
  // integration region. At the end, the maps from all the processes 
  // are added to obtain the final map

  for(int ip=0;ip<nperiodic;ip++){
  for(int jp=0;jp<nperiodic;jp++){
  for(int kp=0;kp<nperiodic;kp++){

    // set corners of octant
    xc1 = xc0 + Periodicity * ( ip - 0.5 );
    yc1 = yc0 + Periodicity * ( jp - 0.5 );
    zc1 = zc0 + Periodicity * ( kp - 0.5 );

    xc2 = xc0 + Periodicity * ( ip + 0.5 );
    yc2 = yc0 + Periodicity * ( jp + 0.5 );
    zc2 = zc0 + Periodicity * ( kp + 0.5 );

    // set corners of slab

    xs1 = x0 + ip*Periodicity + slab_xoffset;
    ys1 = y0 + jp*Periodicity;
    zs1 = z0 + kp*Periodicity;

    xs2 = xs1 + slabsize;
    ys2 = ys1 + BoxSize;
    zs2 = zs1 + BoxSize;
    
    // if slab does not intersect shell, go to next slab
    bb[0][0]=xs1; bb[0][1]=ys1; bb[0][2]=zs1;
    bb[1][0]=xs2; bb[1][1]=ys2; bb[1][2]=zs2;
    if (SlabShellOverlap(bb, rmin, rmax)==0){
	     continue;
    }

    // Note that x and z are switched
    for(int ic=0;ic<Nlocal;ic++){
      if(myid==0) printf("%d / %d for %d,%d,%d\n",ic+1,Nlocal,ip,jp,kp);
      float xL = xs1 + (ic+0.5)*CellSize;
      if(xL>xc1 && xL<xc2){
    for(int jc=0;jc<N;jc++){
      float yL = ys1 + (jc+0.5)*CellSize;
      if(yL>yc1 && yL<yc2){
    for(int kc=0;kc<N;kc++){
      float zL = zs1 + (kc+0.5)*CellSize;
      if(zL>zc1 && zL<zc2){
      
      float r = sqrt(xL*xL + yL*yL + zL*zL);

      if(r    < rmin - 2*CellSize || r > rmax + 2*CellSize) continue;

      float zcur   = Radius2Float(r,Radius2RedshiftTable);

      int index_dv = ic*N*(N+2) + jc*(N+2) + kc;
      
      // CMB Lensing redshift factor
      float kapfac;
      float Wkap;
      if(Parameters.DoMap[KAPCODE]==1){
	       Wkap = Redshift2Float(zcur,Redshift2WKappaTable);
	       kapfac = Wkap * pow(CellSize,3) / pow(r,2) * mapsize / 4. / M_PI;
      }

      // tau redshift factor
      float taufac;
      float Wtau;
      if(Parameters.DoMap[KSZCODE]==1){
	       Wtau = Redshift2Float(zcur,Redshift2WTauTable);	
	       taufac = Wtau * pow(CellSize,3) / pow(r,2) * mapsize / 4. / M_PI;
      }
      
      float D  = growth(zcur,Parameters.Omegam,Parameters.Omegal, Parameters.w)/DInit;
      float D2 = 3. / 7. * D * D;

      long pixel;
      
      float xE,yE,zE;

      // Displacements
      float sx,sy,sz;

      sx = 0; sy = 0; sz = 0;
      if(iLPT > 0) {
	       sx += D * sx1[index_dv];
	       sy += D * sy1[index_dv];
	       sz += D * sz1[index_dv];
      } 
      if(iLPT > 1) {
	       sx += D2 * sx2[index_dv];
	       sy += D2 * sy2[index_dv];
	       sz += D2 * sz2[index_dv];
      }

      // Velocities
      float vx,vy,vz;

      // missing 2lpt for now need to fix
      float aHf = 100*h*pow(pow((1+zcur),3)*Omegam+1-Omegam,0.5)/(1+zcur)/3e5*fofz(Omegam,z);
       
      vx = aHf * sx;
      vy = aHf * sy;
      vz = aHf * sz;

      float vdotn = (vx*xL+vy*yL+vz*zL) / r ;

      // ksz factor v.r/c*tau
      float kszfac;
      if(Parameters.DoMap[KSZCODE]==1) kszfac = taufac * vdotn ;

      xE = xL + sx ;
      yE = yL + sy ; 
      zE = zL + sz ;

      // Subtract contribution from Lagrangian point for lensing convergence
      if(Parameters.DoMap[KAPCODE]==1){
	       vec[2] = xL; vec[1] = yL; vec[0] = zL;
	       vec2pix_nest(Parameters.Nside, vec, &pixel);
	       kapmapl[pixel] -= kapfac ;
      }
      
      // Add contribution to Eulerian point      
      vec[2] = xE; vec[1] = yE; vec[0] = zE; // x-z flipped
      vec2pix_nest(Parameters.Nside, vec, &pixel);

      // Add relative fluctuation weight to cell for Eulerian mode
      float eulerian_signal = 1;
      if(iLPT == 0) eulerian_signal = delta1[index_dv];
      
      if(Parameters.DoMap[KAPCODE]==1) kapmapl[ pixel] += kapfac * eulerian_signal;
      if(Parameters.DoMap[TAUCODE]==1) taumapl[ pixel] += taufac * eulerian_signal;
      if(Parameters.DoMap[KSZCODE]==1) kszmapl[ pixel] += kszfac * eulerian_signal;
      
    }
    }
    }    

    }
    }
    }    

  }
  }
  }
  
  // sum process contributions

  if(Parameters.DoMap[KAPCODE]==1)
    MPI_Reduce(kapmapl, kapmap, mapsize, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
  if(Parameters.DoMap[KSZCODE]==1)
    MPI_Reduce(kszmapl, kszmap, mapsize, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
  if(Parameters.DoMap[TAUCODE]==1)
    MPI_Reduce(taumapl, taumap, mapsize, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
  
  if(myid==0) printf("\n Sum process contributions complete\n");

  MPI_Barrier(MPI_COMM_WORLD);
  double dt = MPI_Wtime() - t1;
  if(myid==0) printf("\n Projection took %le seconds\n",dt);

  // report statistics

  if(Parameters.DoMap[KAPCODE]==1 && myid==0) ReportMapStatistics(kapmap, mapsize," kappa    ");
  if(Parameters.DoMap[KSZCODE]==1 && myid==0) ReportMapStatistics(kszmap, mapsize," kSZ      ");
  if(Parameters.DoMap[TAUCODE]==1 && myid==0) ReportMapStatistics(taumap, mapsize," tau      ");

  if (myid==0) printf("\n Report map stats complete \n");

}

int SlabShellOverlap(float **bb, float rmin, float rmax){

  for(int i1=0;i1<2;i1++){
  for(int j1=0;j1<2;j1++){
  for(int k1=0;k1<2;k1++){

    float rc1 = pow((pow(bb[i1][0],2)+pow(bb[j1][1],2)+pow(bb[k1][2],2)),0.5);

    if(rc1>rmin && rc1<rmax) return 1;

    float dr1ci = rc1 - rmin;
    float dr1co = rc1 - rmax;

    for(int i2=0;i2<2;i2++){
    for(int j2=0;j2<2;j2++){
    for(int k2=0;k2<2;k2++){

      float rc2 = pow((pow(bb[i2][0],2)+pow(bb[j2][1],2)+pow(bb[k2][2],2)),0.5);

      // inner boundary
      float dr2ci = rc2 - rmin;
      if(dr1ci*dr2ci < 0) return 1;

      // outer boundary
      float dr2co = rc2 - rmax;
      if(dr1co*dr2co < 0) return 1;

    }    
    }    
    }    
  }
  }
  }

  return 0;
  
}

void ReportMapStatistics(float *map, int mapsize, char *variable){
  
  int i;
  double  mean, var, rms, sum;
  
  // mean
  sum=0; for(i=0;i<mapsize;i++) sum += map[i];
  mean = sum / mapsize;

  // rms
  sum=0; for(i=0;i<mapsize;i++) sum += pow((map[i]-mean),2);
  var  = sum / mapsize;
  rms  = pow(var,0.5);

  // report
  printf("\n %s mean, rms = %e, %e\n",variable,mean,rms);

}



