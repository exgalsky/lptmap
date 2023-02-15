import numpy as np 
import cosmology as cosmo
import healpy as hp
import matplotlib.pyplot as plt
import scipy.constants as cons
from joblib import Parallel, delayed
from mpi4py import MPI

# MPI communicator initialization
comm = MPI.COMM_WORLD
id = comm.Get_rank()            #number of the process running the code
numProc = comm.Get_size()       #total number of processes running

# Setup cosmology workspace on all processes so that class method functions are initialized
cosmo_wsp = cosmo.cosmology()

# Cube shape is gridx_sz = gridy_sz = gridz_sz, for now, allowing for later change to different x/y/z sizes
gridx_sz = 128 #6144
# gridy_sz = 6144
# gridz_sz = 6144 

# Lattice spacing (a_latt in Websky parlance) in Mpc
gridsz_in_Mpc = 7700 / 6144  # in Mpc; 7700 Mpc box length for websky 6144 cube  

# comoving distance to last scattering in Mpc
comov_lastscatter = 13.8 * 1.e3 # in Mpc

# NSIDE of HEALPix map 
nside = 128

# Effectively \Delta chi, comoving distance interval spacing for LoS integral
geometric_factor = gridsz_in_Mpc**3. / hp.nside2pixarea(nside)

# Path to displacement fields
# path2disp = '/pscratch/sd/m/malvarez/websky-displacements/'
path2disp = '/Users/shamik/Documents/Work/websky_datacube/'



# Run on root process
if id == 0:
# Slice the cube along x which is the slowest varying axis (C indexing)
# Note: all binary files for numpy arrays follow C indexing

    xslab_min = gridx_sz // numProc      # Q
    iter_remains = np.mod(gridx_sz, numProc)    # R 

#  XSZ = R x (Q + 1) + (P-R) x Q 
    xslab_per_Proc = np.zeros((numProc,), dtype=np.int16)  # P = len(zslab_per_Proc)

    xslab_per_Proc[0:int(iter_remains)] = xslab_min + 1     # R procs together get (Q+1)xR z slabs
    xslab_per_Proc[int(iter_remains):]  = xslab_min         # P-R procs together get Qx(P-R) z slabs

    # print(np.sum(zslab_per_Proc))

    del xslab_min

    kappa = np.empty((hp.nside2npix(nside),), dtype=np.float64)
else: 
    kappa = None
    xslab_per_Proc = None 

# Send copy of xslab_per_Proc to each process.
xslab_per_Proc = comm.bcast(xslab_per_Proc, root=0)

# Find the start and stop of the x-axis slab for each process
xslab_start_in_Proc = np.sum(xslab_per_Proc[0:id])    
xslab_stop_in_Proc = xslab_start_in_Proc + xslab_per_Proc[id]

# print(id, zslab_start_in_Proc, zslab_stop_in_Proc)

# Compute single precision offset for reading tranche of displacement values
sp_data_slab_offset = np.sum(xslab_per_Proc[0:id]) * gridx_sz * gridx_sz *  4  # in bytes 

# Setup axes for the slab grid
xaxis = np.arange(xslab_start_in_Proc, xslab_stop_in_Proc, dtype=np.float32)
yaxis = np.arange(0, gridx_sz, dtype=np.float32)
zaxis = np.arange(0, gridx_sz, dtype=np.float32) 

# Setup meshgrid for the sl
grid_qx, grid_qy, grid_qz = np.meshgrid(xaxis, yaxis, zaxis, indexing='ij')
# print(grid_qx.shape, grid_qy.shape, grid_qz.shape)

comov_q = np.sqrt((grid_qx+ 0.5)**2. + (grid_qy+0.5)**2. + (grid_qz+0.5)**2.) * gridsz_in_Mpc

# plt.imshow(comov_q[0,:,:], cmap=plt.cm.Blues, origin='lower')
# plt.title('Initial comoving distance process = '+str(id))
# plt.colorbar()
# plt.savefig('./comov_q_proc'+str(id)+'.png')

redshift_q = cosmo_wsp.comoving_distance2z(comov_q)

# plt.imshow(redshift_q[0,:,:], cmap=plt.cm.Blues, origin='lower')
# plt.title('Lagrangian redshift process = '+str(id))
# plt.colorbar()
# plt.savefig('./redshift_lagrangian_proc'+str(id)+'.png')

growthD = cosmo_wsp.growth_factor_D(redshift_q)

# plt.imshow(growthD[0,:,:], cmap=plt.cm.Blues, origin='lower')
# plt.title('Growth factor D process = '+str(id))
# plt.colorbar()
# plt.savefig('./growthD_proc'+str(id)+'.png')

del xaxis, yaxis, zaxis

grid_sx = np.fromfile(path2disp+'sx1_n128_bin', count=xslab_per_Proc[id] * gridx_sz * gridx_sz, offset=sp_data_slab_offset, dtype=np.float32).reshape(grid_qx.shape) #sx1_7700Mpc_n6144_nb30_nt16
grid_Xx = grid_qx + growthD * grid_sx

del grid_qx, grid_sx

grid_sy = np.fromfile(path2disp+'sy1_n128_bin', count=xslab_per_Proc[id] * gridx_sz * gridx_sz, offset=sp_data_slab_offset, dtype=np.float32).reshape(grid_qy.shape) #sy1_7700Mpc_n6144_nb30_nt16
grid_Xy = grid_qy + growthD * grid_sy

del grid_qy, grid_sy

grid_sz = np.fromfile(path2disp+'sz1_n128_bin', count=xslab_per_Proc[id] * gridx_sz * gridx_sz, offset=sp_data_slab_offset, dtype=np.float32).reshape(grid_qz.shape) #sz1_7700Mpc_n6144_nb30_nt16
grid_Xz = grid_qz + growthD * grid_sz

del grid_qz, grid_sz

del growthD

# comov_X = np.sqrt((grid_Xx+ 0.5)**2. + (grid_Xy+0.5)**2. + (grid_Xz+0.5)**2.) * gridsz_in_Mpc
# plt.imshow(comov_X[0,:,:], cmap=plt.cm.Blues, origin='lower')
# plt.title('Euclidiean comoving distance process = '+str(id))
# plt.colorbar()
# plt.savefig('./comov_X_proc'+str(id)+'.png')

# exit()

ipix_grid = hp.vec2pix(nside, grid_Xx.flatten(), grid_Xy.flatten(), grid_Xz.flatten())
ipix_grid = ipix_grid.reshape((xslab_per_Proc[id], gridx_sz, gridx_sz))

# plt.imshow(ipix_grid[0,:,:], cmap=plt.cm.rainbow, origin='lower')
# plt.title('HEALPix pixels process = '+str(id))
# plt.colorbar()
# plt.savefig('./HEALPix-pixel_proc'+str(id)+'.png')

# exit()
# normalization factors are TBD
lensing_kernel_grid = geometric_factor * (3./2.) * cosmo_wsp.params['Omega_m'] * (cosmo_wsp.params['h'] * 100. * cons.kilo / cons.c )**2. * (1 + redshift_q) * (1. - (comov_q/comov_lastscatter)) / comov_q
mask_comov = comov_q <= np.max(comov_q[0,0,:])

del redshift_q, comov_q

ipix_unique = np.unique(ipix_grid)

def LoS_integration4hpx(kernel, comov_mask):
    return np.sum(kernel[comov_mask])

def call_LoSinteg(ipix):
    sel = np.where(ipix_grid.flatten() == ipix)[0]
    return LoS_integration4hpx(lensing_kernel_grid.flatten()[sel], mask_comov.flatten()[sel])

# kappa_values = Parallel(n_jobs=-1, prefer="threads")(delayed (call_LoSinteg)(pix) for pix in ipix_unique)
kappa_values = []
for pix in ipix_unique:
    kappa_values.append(call_LoSinteg(pix))

del lensing_kernel_grid, mask_comov

kappa_slab = np.zeros(hp.nside2npix(nside))
kappa_slab[ipix_unique] = kappa_values

comm.Reduce([kappa_slab, MPI.DOUBLE], kappa, op=MPI.SUM, root=0)

del kappa_slab

if id == 0:
    # hp.savefig('./kappa-map_websky1lpt_nside'+str(nside)+'.fits', kappa, dtype=np.float64)
    hp.orthview(kappa, rot=[0.,90.,0.], cmap='inferno', half_sky=True, title=r'$\kappa$ map')
    hp.graticule(ls='-', lw=0.25, c='w')
    plt.savefig('./kappa_map_smallcube.png', bbox_inches='tight', pad_inches=0.1)








