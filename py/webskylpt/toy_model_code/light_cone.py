import numpy as np 
import cosmology as cosmo
import healpy as hp
import matplotlib.pyplot as plt
import scipy.constants as cons
from time import time

run_with_mpi = False
if run_with_mpi:
    try: 
        from mpi4py import MPI
    except:
        print("WARNING: mpi4py not found, fallback to serial implementation.")  # todo: Replace print warning messages with proper logging implementation
        run_with_mpi = False

# MPI communicator initialization
if run_with_mpi:
    comm = MPI.COMM_WORLD
    id = comm.Get_rank()            #number of the process running the code
    numProc = comm.Get_size()       #total number of processes running
else:
    id = 0
    numProc = 1 

t0 = time()
# Setup cosmology workspace on all processes so that class method functions are initialized
cosmo_wsp = cosmo.cosmology()

t1 = time()

print("Evaluation of cosmology took", t1-t0, "s for Proc ", id)    # todo: Timing statements should also be implemented with logging module
# Cube shape is grid_xsize = gridy_sz = gridz_sz, for now, allowing for later change to different x/y/z sizes
grid_xsize = 128 #6144
# grid_ysize = 6144
# grid_zsize = 6144 

L_box = 7700. # in Mpc

# Lattice spacing (a_latt in Websky parlance) in Mpc
gridsz_in_Mpc = L_box / 6144 #grid_xsize  # in Mpc; 7700 Mpc box length for websky 6144 cube  

# comoving distance to last scattering in Mpc
comov_lastscatter = 13.8 * (cons.giga / cons.mega) # in Mpc

# NSIDE of HEALPix map 
nside = 128
npix = hp.nside2npix(nside)

# Effectively \Delta chi, comoving distance interval spacing for LoS integral
geometric_factor = gridsz_in_Mpc**3. / hp.nside2pixarea(nside)

# Path to displacement fields
# path2disp = '/pscratch/sd/m/malvarez/websky-displacements/'
path2disp = '/Users/shamik/Documents/Work/websky_datacube/'


# Run on root process
if id == 0:
# Slice the cube along x which is the slowest varying axis (C indexing)
# Note: all binary files for numpy arrays follow C indexing

    xslab_min = grid_xsize // numProc      # Q
    iter_remains = np.mod(grid_xsize, numProc)    # R 

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
if run_with_mpi:
    xslab_per_Proc = comm.bcast(xslab_per_Proc, root=0)

# Find the start and stop of the x-axis slab for each process
xslab_start_in_Proc = np.sum(xslab_per_Proc[0:id])    
xslab_stop_in_Proc = xslab_start_in_Proc + xslab_per_Proc[id]

# print(id, zslab_start_in_Proc, zslab_stop_in_Proc)

# Compute single precision offset for reading tranche of displacement values
sp_data_slab_offset = np.sum(xslab_per_Proc[0:id]) * grid_xsize * grid_xsize *  4  # in bytes 

t2 = time()

print("Slab decomposition took", t2-t1, "s for Proc ", id)

# Setup axes for the slab grid
xaxis = np.arange(xslab_start_in_Proc, xslab_stop_in_Proc, dtype=np.int16)
yaxis = np.arange(0, grid_xsize, dtype=np.int16)
zaxis = np.arange(0, grid_xsize, dtype=np.int16) 

# Setup meshgrid for the slab 
grid_qx, grid_qy, grid_qz = np.meshgrid(xaxis, yaxis, zaxis, indexing='ij') # 40.5
# print(grid_qx.shape, grid_qy.shape, grid_qz.shape)

del xaxis, yaxis, zaxis

t3 = time()
print("Grid setup took", t3-t2, "s for Proc ", id)
# Lagrangian comoving distance grid for the slab
comov_q = (np.sqrt((grid_qx+ 0.5)**2. + (grid_qy+0.5)**2. + (grid_qz+0.5)**2.) * gridsz_in_Mpc).astype(np.float32) # 27 

# plt.imshow(comov_q[0,:,:], cmap=plt.cm.Blues, origin='lower')
# plt.title('Initial comoving distance process = '+str(id))
# plt.colorbar()
# plt.savefig('./comov_q_proc'+str(id)+'.png')

t4 = time()
print("Comoving distance q took", t4-t3, "s for Proc ", id)

# Lagrangian redshift grid for the slab
redshift_q = cosmo_wsp.comoving_distance2z(comov_q).astype(np.float32)  # 27

# plt.imshow(redshift_q[0,:,:], cmap=plt.cm.Blues, origin='lower')
# plt.title('Lagrangian redshift process = '+str(id))
# plt.colorbar()
# plt.savefig('./redshift_lagrangian_proc'+str(id)+'.png')

t5 = time()
print("Redshift took", t5-t4, "s for Proc ", id)

# Growth factor D grid for the slab
growthD = cosmo_wsp.growth_factor_D(redshift_q).astype(np.float32)  # 27

# plt.imshow(growthD[0,:,:], cmap=plt.cm.Blues, origin='lower')
# plt.title('Growth factor D process = '+str(id))
# plt.colorbar()
# plt.savefig('./growthD_proc'+str(id)+'.png')

t6 = time()
print("Growth factor took", t6-t5, "s for Proc ", id)

# Compute Euclidean grids. Cleanup unwanted grids

grid_sx = np.fromfile(path2disp+'sx1_n128_bin', count=xslab_per_Proc[id] * grid_xsize * grid_xsize, offset=sp_data_slab_offset, dtype=np.float32).reshape(grid_qx.shape) #sx1_7700Mpc_n6144_nb30_nt16
grid_Xx = (grid_qx + growthD * grid_sx).flatten()

t7 = time()
print("Displacement x I/O and Eulerian grid x took", t7-t6, "s for Proc ", id)

del grid_qx, grid_sx

grid_sy = np.fromfile(path2disp+'sy1_n128_bin', count=xslab_per_Proc[id] * grid_xsize * grid_xsize, offset=sp_data_slab_offset, dtype=np.float32).reshape(grid_qy.shape) #sy1_7700Mpc_n6144_nb30_nt16
grid_Xy =(grid_qy + growthD * grid_sy).flatten()

t8 = time()
print("Displacement y I/O and Eulerian grid y took", t8-t7, "s for Proc ", id)

del grid_qy, grid_sy

grid_sz = np.fromfile(path2disp+'sz1_n128_bin', count=xslab_per_Proc[id] * grid_xsize * grid_xsize, offset=sp_data_slab_offset, dtype=np.float32).reshape(grid_qz.shape) #sz1_7700Mpc_n6144_nb30_nt16
grid_Xz = (grid_qz + growthD * grid_sz).flatten()

t9 = time()
print("Displacement z I/O and Eulerian grid z took", t9-t8, "s for Proc ", id)

del grid_qz, grid_sz

del growthD

# comov_X = np.sqrt((grid_Xx+ 0.5)**2. + (grid_Xy+0.5)**2. + (grid_Xz+0.5)**2.) * gridsz_in_Mpc
# plt.imshow(comov_X[0,:,:], cmap=plt.cm.Blues, origin='lower')
# plt.title('Euclidiean comoving distance process = '+str(id))
# plt.colorbar()
# plt.savefig('./comov_X_proc'+str(id)+'.png')

# Compute healpix pixel grid from Euclidean x, y, z values
ipix_grid = hp.vec2pix(nside, grid_Xx, grid_Xy, grid_Xz)
ipix_grid = ipix_grid.reshape((xslab_per_Proc[id], grid_xsize, grid_xsize))

del grid_Xx, grid_Xy, grid_Xz

t10 = time()
print("HPX pixel grid took", t10-t9, "s for Proc ", id)

# plt.imshow(ipix_grid[0,:,:], cmap=plt.cm.rainbow, origin='lower')
# plt.title('HEALPix pixels process = '+str(id))
# plt.colorbar()
# plt.savefig('./HEALPix-pixel_proc'+str(id)+'.png')

# normalization factors fixed! 14/02/2023 
# W_kappa = a_latt^3 / Omega_pix * (3/2) * Omega_M * (H_0 / c_in_km_per_s)^2 * (1 + z) *(1 - X/X_*) / X
lensing_kernel_grid = geometric_factor * (3./2.) * cosmo_wsp.params['Omega_m'] * (cosmo_wsp.params['h'] * 100. * cons.kilo / cons.c )**2. * (1 + redshift_q) * (1. - (comov_q/comov_lastscatter)) / comov_q

del redshift_q
t11 = time()
print("Lensing kernel grid took", t11-t10, "s for Proc ", id)

# Mask limiting to constant radius in the cube with origin at (0,0,0) corner.
mask_comov = np.ones(comov_q.shape)
mask_comov[comov_q > L_box] = 0.

del comov_q

# Find only the unique pixels to compute over
# ipix_unique = np.unique(ipix_grid)

# LoS integral sum function definitions:
# def LoS_integration4hpx(kernel, comov_mask):
#     return np.sum(kernel[comov_mask])

# def call_LoSinteg(ipix):
#     sel = np.where(ipix_grid.flatten() == ipix)[0]
#     return LoS_integration4hpx(lensing_kernel_grid.flatten()[sel], mask_comov.flatten()[sel])

# # Compute on all available threads for each process
# kappa_values = Parallel(n_jobs=-1, prefer="threads")(delayed (call_LoSinteg)(pix) for pix in ipix_unique)


kappa_slab, edges = np.histogram(ipix_grid, bins=npix, range=(-0.5,npix-0.5), weights=lensing_kernel_grid*mask_comov, density=False)

t12 = time()
print("LoS integral by histogram took", t7-t6, "s for Proc ", id)

del lensing_kernel_grid, edges, mask_comov

# Map the part LoS integrals to relevant pixels 
# kappa_slab = np.zeros(hp.nside2npix(nside))
# kappa_slab[ipix_unique] = kappa_values

# del kappa_values, ipix_unique

# Combine intergals for slabs with reduction
if run_with_mpi:
    comm.Reduce([kappa_slab, MPI.DOUBLE], kappa, op=MPI.SUM, root=0)
else:
    kappa = kappa_slab

t13 = time()
print("Reduce kappa took", t13-t12, "s for Proc ", id)
del kappa_slab

if id == 0:
    t14 = time()
    print("Job completion took", t14-t0, "s for Proc ", id)
    # Save map and plot figure:
    # hp.write_map('../output/kappa-map_websky1lpt_no768_nside'+str(nside)+'_MPI.fits', kappa, dtype=np.float64, overwrite=True)
    hp.orthview(kappa, rot=[0.,90.,0.], cmap='inferno', half_sky=True, title=r'$\kappa$ map')
    hp.graticule(ls='-', lw=0.25, c='w')
    plt.savefig('./kappa_small_cube_128.png', bbox_inches='tight', pad_inches=0.1) #../output/kappa_map_fullcube_no768.png








