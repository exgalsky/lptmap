import numpy as np 
import cosmology as cosmo
import healpy as hp
import jax_healpix as jhp
import matplotlib.pyplot as plt
import scipy.constants as cons
# from mpi4py import MPI
# import mpi4jax as mjx
from functools import partial
import jax
import jax.numpy as jnp
from time import time

t0 = time()
# Setup cosmology workspace on all processes so that class method functions are initialized
cosmo_wsp = cosmo.cosmology()

t1 = time()

print("Evalutation of cosmology took", t1-t0, "s ")
# Cube shape is parameterized by grid_nside and actual physical size of the box size of the simulation. Changing these control the lattice spacing
grid_nside = 768 #6144
L_box = 7700. # in Mpc

# Lattice spacing (a_latt in Websky parlance) in Mpc
lattice_size_in_Mpc = L_box / grid_nside  # in Mpc; 7700 Mpc box length for websky 6144 cube  

# comoving distance to last scattering in Mpc
comov_lastscatter = 13.8 * (cons.giga / cons.mega) # in Mpc

# NSIDE of HEALPix map 
nside = 1024
npix = hp.nside2npix(nside)
solidang_pix = 4*np.pi / npix

# Effectively \Delta chi, comoving distance interval spacing for LoS integral
geometric_factor = lattice_size_in_Mpc**3. / solidang_pix

# Path to displacement fields
# path2disp = '/pscratch/sd/m/malvarez/websky-displacements/'
path2disp = '/Users/shamik/Documents/Work/websky_datacube/'

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
    return q_i * Dgrid_in_Mpc + growth_i * s_i + 0.5 + trans*Dgrid_in_Mpc

@jax.jit
def lensing_kernel_F(comov_q_i, redshift_i):
    return geometric_factor * (3./2.) * cosmo_wsp.params['Omega_m'] * (cosmo_wsp.params['h'] * 100. * cons.kilo / cons.c )**2. * (1 + redshift_i) * (1. - (comov_q_i/comov_lastscatter)) / comov_q_i

for translation in origin_shift:
    print(translation)
    t3_5 = time()
    lagrange_grid = jax.vmap(comoving_q, in_axes=(0, 0, 0, None, None), out_axes=0)(grid_qx, grid_qy, grid_qz, translation, lattice_size_in_Mpc)

    t4 = time()
    print("Lagrange grid took", t4-t3_5, "s ")
    redshift_grid = jax.vmap(cosmo_wsp.comoving_distance2z)(lagrange_grid)

    t5 = time()
    print("Redshift took", t5-t4, "s ")

    growth_grid = jax.vmap(cosmo_wsp.growth_factor_D)(redshift_grid)

    t6 = time()
    print("Growth took", t6-t5, "s ")
    # print(lagrange_grid.shape, redshift_grid.shape, growth_grid.shape)

    grid_sx = jnp.asarray(np.fromfile(path2disp+'sx1_7700Mpc_n6144_nb30_nt16_no768', count=grid_nside * grid_nside * grid_nside, dtype=jnp.float32).reshape(grid_qx.shape))
    grid_Xx = jax.vmap(euclid_i, in_axes=(0, 0, 0, None, None), out_axes=0)(grid_qx, grid_sx, growth_grid, lattice_size_in_Mpc, translation[0])
    del grid_sx

    grid_sy = jnp.asarray(np.fromfile(path2disp+'sy1_7700Mpc_n6144_nb30_nt16_no768', count=grid_nside * grid_nside * grid_nside, dtype=jnp.float32).reshape(grid_qy.shape))
    grid_Xy = jax.vmap(euclid_i, in_axes=(0, 0, 0, None, None), out_axes=0)(grid_qy, grid_sy, growth_grid, lattice_size_in_Mpc, translation[1])
    del grid_sy

    grid_sz = jnp.asarray(np.fromfile(path2disp+'sz1_7700Mpc_n6144_nb30_nt16_no768', count=grid_nside * grid_nside * grid_nside, dtype=jnp.float32).reshape(grid_qz.shape))
    grid_Xz = jax.vmap(euclid_i, in_axes=(0, 0, 0, None, None), out_axes=0)(grid_qz, grid_sz, growth_grid, lattice_size_in_Mpc, translation[2])
    del grid_sz

    # plt.imshow(np.asarray(grid_Xx)[grid_nside-1,:,:], cmap=plt.cm.Blues, origin='lower')
    # plt.title('Euclidean grid x coord')
    # plt.colorbar()
    # plt.savefig('../output/euclid_x_z0plane.png')
    # plt.close()

    # plt.imshow(np.asarray(grid_Xy)[:,grid_nside-1,:], cmap=plt.cm.Blues, origin='lower')
    # plt.title('Euclidean grid y coord')
    # plt.colorbar()
    # plt.savefig('../output/euclid_y_x0plane.png')
    # plt.close()

    # plt.imshow(np.asarray(grid_Xz)[:,:,grid_nside-1], cmap=plt.cm.Blues, origin='lower')
    # plt.title('Euclidean grid z coord')
    # plt.colorbar()
    # plt.savefig('../output/euclid_z_y0plane.png')
    # plt.close()

    t7 = time()
    print("Euclid grid took", t7-t6, "s ")


    # Compute healpix pixel grid from Euclidean x, y, z values
    ipix_grid = jhp.vec2pix(nside, grid_Xx, grid_Xy, grid_Xz)
    del grid_Xx, grid_Xy, grid_Xz

    t8 = time()
    print("HPX pixel grid took", t8-t7, "s ")

    # kernel_grid = 
    kernel_sphere = jnp.where(lagrange_grid <= L_box, jax.vmap(lensing_kernel_F)(lagrange_grid, redshift_grid), 0.)

    t9 = time()
    print("Kernel grid took", t9-t8, "s ")

    hpxmap, edges = jnp.histogram(ipix_grid, bins=npix, range=(-0.5,npix-0.5), weights=kernel_sphere, density=False)

    skymap += np.asarray(hpxmap)

    del hpxmap, edges

    t10 = time()
    print("Project to healpix took", t10-t9, "s ")

print("Job completion took", t10-t0, "s ")
# Save map and plot figure:
hp.write_map('./output/kappa-map_websky1lpt_nside'+str(nside)+'_768.fits', skymap, dtype=np.float64, overwrite=True)
fig = plt.figure(figsize=(6,4), dpi=600)
hp.mollview(skymap, cmap=plt.cm.Spectral_r, min=0., max=2., title=r'$\kappa$ map', fig=fig.number, xsize=3000)
hp.graticule(ls='-', lw=0.25, c='k')
plt.savefig('./output/kappa_map_jax_768.png', bbox_inches='tight', pad_inches=0.1)