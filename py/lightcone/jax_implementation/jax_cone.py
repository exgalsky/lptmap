import numpy as np 
import cosmology as cosmo
import healpy as hp
import jax_healpix as jhp
import matplotlib.pyplot as plt
import scipy.constants as cons
import os, sys
# from mpi4py import MPI
# import mpi4jax as mjx
from functools import partial
import jax
import jax.numpy as jnp
from time import time

# ------ hardcoded parameters

grid_nside            = 768   # cube shape is parameterized by grid_nside; full resolution for websky is 6144
L_box                 = 7700. # periodic box size in comoving Mpc; lattice spacing is L_box / grid_nside
comov_lastscatter_Gpc = 13.8  # conformal distance to last scattering surface in Gpc
zmin                  = 0.05  # minimum redshift for projection (=0.05 for websky products)
zmax                  = 4.5   # maximum redshift for projection (=4.50 for websky products)

# Paths to displacement fields
try:
    path2disp = os.environ['LPT_DISPLACEMENTS_PATH']
except:
    print("LPT_DISPLACEMENTS_PATH not set, exiting...")
    sys.exit(1)
sxfile = path2disp+'sx1_7700Mpc_n6144_nb30_nt16_no768'
syfile = path2disp+'sy1_7700Mpc_n6144_nb30_nt16_no768'
szfile = path2disp+'sz1_7700Mpc_n6144_nb30_nt16_no768'

# ------ end hardcoded parameters

t0 = time()
# Setup cosmology workspace on all processes so that class method functions are initialized
cosmo_wsp = cosmo.cosmology()

t1 = time()

print("Evalutation of cosmology took", t1-t0, "s ")

# Lattice spacing (a_latt in Websky parlance) in Mpc
lattice_size_in_Mpc = L_box / grid_nside  # in Mpc; 7700 Mpc box length for websky 6144 cube  

# comoving distance to last scattering in Mpc
comov_lastscatter = comov_lastscatter_Gpc * (cons.giga / cons.mega) # in Mpc

# minimum and maximum radii of projection
chimin = cosmo_wsp.comoving_distance(zmin)
chimax = cosmo_wsp.comoving_distance(zmax)

print("chimin, chimax: ",chimin,chimax)
# NSIDE of HEALPix map 
nside = 1024
npix = hp.nside2npix(nside)
solidang_pix = 4*np.pi / npix

# Effectively \Delta chi, comoving distance interval spacing for LoS integral
geometric_factor = lattice_size_in_Mpc**3. / solidang_pix

t2 = time()
# Setup axes for the slab grid
xaxis = jnp.arange(0, grid_nside, dtype=jnp.int16)
yaxis = jnp.arange(0, grid_nside, dtype=jnp.int16) 
zaxis = jnp.arange(0, grid_nside, dtype=jnp.int16) 

# Setup meshgrid for the slab 
grid_qx, grid_qy, grid_qz = jnp.meshgrid(xaxis, yaxis, zaxis, indexing='ij')

del xaxis, yaxis, zaxis

skymap = np.zeros((npix,))
shift_param = grid_nside
origin_shift = [(0,0,0), (-shift_param,0,0), (0,-shift_param,0), (-shift_param,-shift_param,0),
                (0,0,-shift_param), (-shift_param,0,-shift_param), (0,-shift_param,-shift_param), (-shift_param,-shift_param,-shift_param)]
t3 = time()
print("Grid setup took", t3-t2, "s ")

# Lagrangian comoving distance grid for the slab

@partial(jax.jit, static_argnames=['trans_vec', 'Dgrid_in_Mpc'])
def comoving_q(x_i, y_i, z_i, trans_vec, Dgrid_in_Mpc):
    return jnp.sqrt((x_i + 0.5 + trans_vec[0])**2. + (y_i + 0.5 + trans_vec[1])**2. + (z_i + 0.5 + trans_vec[2])**2.) * Dgrid_in_Mpc

@partial(jax.jit, static_argnames=['Dgrid_in_Mpc', 'trans'])
def euclid_i(q_i, s_i, growth_i, Dgrid_in_Mpc, trans):
    return (q_i + 0.5 + trans) * Dgrid_in_Mpc + growth_i * s_i

@jax.jit
def lensing_kernel_F(comov_q_i, redshift_i):
    return geometric_factor * (3./2.) * cosmo_wsp.params['Omega_m'] * (cosmo_wsp.params['h'] * 100. * cons.kilo / cons.c )**2. * (1 + redshift_i) * (1. - (comov_q_i/comov_lastscatter)) / comov_q_i

def read_displacement(filename):
    return jnp.asarray(np.fromfile(filename, count=grid_nside * grid_nside * grid_nside, dtype=jnp.float32).reshape(grid_qx.shape), dtype=jnp.float32)

t3b = time()
print("Jit compilation took", t3b-t3, "s ")
t3 = t3b

store_displacements=True
if store_displacements:
    grid_sx = read_displacement(sxfile)
    grid_sy = read_displacement(syfile)
    grid_sz = read_displacement(szfile)

t3b = time()
print("I/O took", t3b-t3, "s ")
t3 = t3b

for translation in origin_shift:
    print(translation)
    t3_5 = time()
    lagrange_grid = jax.vmap(comoving_q, in_axes=(0, 0, 0, None, None), out_axes=0)(grid_qx, grid_qy, grid_qz, translation, lattice_size_in_Mpc)

    t4 = time() ; print("Lagrange grid took", t4-t3_5, "s ")
    redshift_grid = jax.vmap(cosmo_wsp.comoving_distance2z)(lagrange_grid)

    t5 = time() ; print("Redshift took", t5-t4, "s ")

    growth_grid = jax.vmap(cosmo_wsp.growth_factor_D)(redshift_grid)

    t6 = time() ; print("Growth took", t6-t5, "s ")

    if not store_displacements: grid_sx = read_displacement(sxfile)
    grid_Xx = jax.vmap(euclid_i, in_axes=(0, 0, 0, None, None), out_axes=0)(grid_qx, grid_sx, growth_grid, lattice_size_in_Mpc, translation[0])
    if not store_displacements: del grid_sx

    if not store_displacements: grid_sy = read_displacement(syfile)
    grid_Xy = jax.vmap(euclid_i, in_axes=(0, 0, 0, None, None), out_axes=0)(grid_qy, grid_sy, growth_grid, lattice_size_in_Mpc, translation[1])
    if not store_displacements: del grid_sy

    if not store_displacements: grid_sz = read_displacement(szfile)
    grid_Xz = jax.vmap(euclid_i, in_axes=(0, 0, 0, None, None), out_axes=0)(grid_qz, grid_sz, growth_grid, lattice_size_in_Mpc, translation[2])
    if not store_displacements: del grid_sz

    t7 = time() ; print("Displacements took", t7-t6, "s ")

    # Compute healpix pixel grid from Euclidean x, y, z values
    ipix_grid = jhp.vec2pix(nside, grid_Xz, grid_Xy, grid_Xx)
    del grid_Xx, grid_Xy, grid_Xz

    t8 = time() ; print("HPX pixel grid (Eulerian) took", t8-t7, "s ")

    kernel_sphere = jnp.where((lagrange_grid >= chimin) & (lagrange_grid <= chimax), jax.vmap(lensing_kernel_F)(lagrange_grid, redshift_grid), 0.)

    t9 = time() ; print("Kernel grid (Eulerian) took", t9-t8, "s ")

    skymap += np.asarray(jnp.histogram(ipix_grid, bins=npix, range=(-0.5,npix-0.5), weights=kernel_sphere, density=False)[0])
    del ipix_grid, kernel_sphere

    t10 = time() ; print("Project to healpix (Eulerian) took", t10-t9, "s ")

    # Compute healpix pixel grid from Lagrangian x, y, z values
    ipix_grid = jhp.vec2pix(nside, (grid_qz + 0.5 + translation[2])*lattice_size_in_Mpc, (grid_qy + 0.5 + translation[1])*lattice_size_in_Mpc, (grid_qx + 0.5 + translation[0])*lattice_size_in_Mpc)

    t11 = time() ; print("HPX pixel grid (Lagrangian) took", t11-t10, "s ")

    kernel_sphere = jnp.where((lagrange_grid >= chimin) & (lagrange_grid <= chimax), jax.vmap(lensing_kernel_F)(lagrange_grid, redshift_grid), 0.)

    t12 = time() ; print("Kernel grid (Lagrangian) took", t12-t11, "s ")

    skymap += np.asarray(jnp.histogram(ipix_grid, bins=npix, range=(-0.5,npix-0.5), weights=-kernel_sphere, density=False)[0])
    del lagrange_grid, kernel_sphere, ipix_grid

    t13 = time()
    print("Project to healpix (Lagrangian) took", t13-t12, "s ")

del grid_qx, grid_qy, grid_qz

print("Job completion took", t13-t0, "s ")
# Save map and plot figure:
hp.write_map('./output/kappa-map_websky1lpt_nside'+str(nside)+'_768.fits', skymap, dtype=np.float64, overwrite=True)
fig = plt.figure(figsize=(6,4), dpi=600)
hp.mollview(skymap, cmap=plt.cm.Spectral_r, min=0., max=2., title=r'$\kappa$ map', fig=fig.number, xsize=3000)
hp.graticule(ls='-', lw=0.25, c='k')
plt.savefig('./output/kappa_map_jax_768_holesremoved.png', bbox_inches='tight', pad_inches=0.1)