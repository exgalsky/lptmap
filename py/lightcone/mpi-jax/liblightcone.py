import numpy as np 
import healpy as hp
import jax_healpix as jhp
import kernel_lib as kl
from functools import partial
import jax
import jax.numpy as jnp
from tqdm import tqdm



def _read_displacement(filename, chunk_shape, chunk_offset):
    return np.fromfile(filename, count=chunk_shape[0] * chunk_shape[1] * chunk_shape[2], offset=chunk_offset, dtype=jnp.float32).reshape(chunk_shape)

class lightcone_workspace():
    def __init__(self, cosmo_workspace, grid_nside, map_nside, box_length_in_Mpc, zmin, zmax):
        self.grid_nside = grid_nside 
        self.map_nside = map_nside
        self.npix = hp.nside2npix(self.map_nside)
        self.L_box = box_length_in_Mpc
        self.cosmo = cosmo_workspace
        self.chimin = self.cosmo.comoving_distance(zmin)
        self.chimax = self.cosmo.comoving_distance(zmax)
        
    @partial(jax.jit, static_argnames=['self', 'grid_xstarts', 'grid_xstops', 'grid_ystarts', 'grid_ystops', 'grid_zstarts', 'grid_zstops'])
    def grid2map(self, grid_sx, grid_sy, grid_sz, grid_xstarts, grid_xstops, grid_ystarts, grid_ystops, grid_zstarts, grid_zstops):
        # Lattice spacing (a_latt in Websky parlance) in Mpc
        lattice_size_in_Mpc = self.L_box / self.grid_nside

        solidang_pix = 4*np.pi / self.npix

        # Effectively \Delta chi, comoving distance interval spacing for LoS integral
        geometric_factor = lattice_size_in_Mpc**3. / solidang_pix

        # Setup axes for the slab grid
        xaxis = jnp.arange(grid_xstarts, grid_xstops, dtype=jnp.int16)
        yaxis = jnp.arange(grid_ystarts, grid_ystops, dtype=jnp.int16)
        zaxis = jnp.arange(grid_zstarts, grid_zstops, dtype=jnp.int16)

        skymap = jnp.zeros((self.npix,), dtype=jnp.float64)    

        shift_param = self.grid_nside
        origin_shift = [(0,0,0), (-shift_param,0,0), (0,-shift_param,0), (-shift_param,-shift_param,0),
                        (0,0,-shift_param), (-shift_param,0,-shift_param), (0,-shift_param,-shift_param), (-shift_param,-shift_param,-shift_param)]

        # Lagrangian comoving distance grid for the slab
        # @partial(jax.jit, static_argnames=['trans_vec', 'Dgrid_in_Mpc'])
        def lagrange_mesh(x_axis, y_axis, z_axis, trans_vec, Dgrid_in_Mpc):
            return jnp.meshgrid( jnp.float32((x_axis + 0.5 + trans_vec[0]) * Dgrid_in_Mpc), jnp.float32((y_axis + 0.5 + trans_vec[1]) * Dgrid_in_Mpc), jnp.float32((z_axis + 0.5 + trans_vec[2]) * Dgrid_in_Mpc), indexing='ij')

        # @jax.jit
        def comoving_q(x_i, y_i, z_i):
            return jnp.sqrt(x_i**2. + y_i**2. + z_i**2.).astype(jnp.float32)

        # @partial(jax.jit, static_argnames=['Dgrid_in_Mpc', 'trans'])
        def euclid_i(q_i, s_i, growth_i, Dgrid_in_Mpc, trans):
            return (q_i + growth_i * s_i).astype(jnp.float32)
            

        for translation in origin_shift:

            # t4 = time()

            grid_qx, grid_qy, grid_qz = lagrange_mesh(xaxis, yaxis, zaxis, translation, lattice_size_in_Mpc)

            # t5 = time() ; print("Largrangian meshgrid took", t5 - t4, "s ")

            lagrange_grid = jax.vmap(comoving_q, in_axes=(0, 0, 0), out_axes=0)(grid_qx, grid_qy, grid_qz)    # 4 : 22

            # t6 = time() ; print("Lagrangian comoving distance grid took", t6 - t5, "s ")
            redshift_grid = jax.vmap(self.cosmo.comoving_distance2z)(lagrange_grid) 

            # t7 = time() ; print("Redshift took", t7-t6, "s ")
            # Compute healpix pixel grid from Lagrangian x, y, z values
            ipix_grid = jhp.vec2pix(self.map_nside, grid_qz, grid_qy, grid_qx)

            # t8 = time() ; print("HPX pixel grid (Lagrangian) took", t8-t7, "s ")

            kernel_sphere = jnp.where((lagrange_grid >= self.chimin) & (lagrange_grid <= self.chimax), jax.vmap(kl.lensing_kernel_F, in_axes=(None, None, 0, 0), out_axes=0 )(self.cosmo, geometric_factor, lagrange_grid, redshift_grid), 0.)

            # t9 = time() ; print("Kernel grid (Lagrangian) took", t9-t8, "s ")

            skymap = skymap.at[:self.npix].add(jnp.histogram(ipix_grid, bins=self.npix, range=(-0.5, self.npix-0.5), weights=-kernel_sphere, density=False)[0])      
            del kernel_sphere, ipix_grid         

            # t10 = time() ; print("Project to healpix (Lagrangian) took", t10-t9, "s ")

            growth_grid = jax.vmap(self.cosmo.growth_factor_D)(redshift_grid)
        
            # t11 = time() ; print("Growth took", t11-t10, "s ")

            grid_Xx = jax.vmap(euclid_i, in_axes=(0, 0, 0, None, None), out_axes=0)(grid_qx, grid_sx, growth_grid, lattice_size_in_Mpc, translation[0])
            del grid_qx

            grid_Xy = jax.vmap(euclid_i, in_axes=(0, 0, 0, None, None), out_axes=0)(grid_qy, grid_sy, growth_grid, lattice_size_in_Mpc, translation[1])
            del grid_qy

            grid_Xz = jax.vmap(euclid_i, in_axes=(0, 0, 0, None, None), out_axes=0)(grid_qz, grid_sz, growth_grid, lattice_size_in_Mpc, translation[2])
            del grid_qz, growth_grid

            # t12 = time() ; print("Displacements took", t12-t11, "s ")

            # Compute healpix pixel grid from Euclidean x, y, z values
            ipix_grid = jhp.vec2pix(self.map_nside, grid_Xz, grid_Xy, grid_Xx)
            del grid_Xx, grid_Xy, grid_Xz               

            # t13 = time() ; print("HPX pixel grid (Eulerian) took", t13-t12, "s ")

            kernel_sphere = jnp.where((lagrange_grid >= self.chimin) & (lagrange_grid <= self.chimax), jax.vmap(kl.lensing_kernel_F, in_axes=(None, None, 0, 0), out_axes=0 )(self.cosmo, geometric_factor, lagrange_grid, redshift_grid), 0.) 
            del lagrange_grid, redshift_grid

            # t14 = time() ; print("Kernel grid (Eulerian) took", t14-t13, "s ")

            skymap = skymap.at[:self.npix].add(jnp.histogram(ipix_grid, bins=self.npix, range=(-0.5, self.npix-0.5), weights=kernel_sphere, density=False)[0])
            del ipix_grid, kernel_sphere       

            # t15 = time() ; print("Project to healpix (Eulerian) took", t15-t14, "s ")

        del grid_sx, grid_sy, grid_sz

        return skymap
    
    def lpt2map(self, dispfilenames, backend, bytes_per_cell=4):        #kernel_list, 

        data_shape = (self.grid_nside, self.grid_nside, self.grid_nside)
        # HARDCODED PARAMETERS -- NEED TO DOCUMENT AND IMPLEMENT USER SETTING AT RUNTIME
        # jax reports 73 GB for 768^3 on Perlmutter; accounting for an overhead of 1.5, this is peak_per_cell_memory = 115
        peak_per_cell_memory = 150.0
        jax_overhead_factor  = 1.5
        backend.datastream_setup(data_shape, bytes_per_cell, peak_per_cell_memory, jax_overhead_factor, decom_type='slab', divide_axis=0)
        jax_iterator = backend.get_iterator()
        obs_map = np.zeros((self.npix,))

        for iter in tqdm(jax_iterator, ncols=120):

            print(f"start, stop, offset, shape: { iter }")
            
            grid_sx = _read_displacement(dispfilenames[0], iter[3], iter[2])
            grid_sy = _read_displacement(dispfilenames[1], iter[3], iter[2])
            grid_sz = _read_displacement(dispfilenames[2], iter[3], iter[2])

            obs_map += np.array(self.grid2map(jnp.asarray(grid_sx), jnp.asarray(grid_sy), jnp.asarray(grid_sz), iter[0], iter[1], 0, self.grid_nside, 0, self.grid_nside), dtype=np.float64)  #, kernel_list

        return backend.mpi_backend.reduce2map(obs_map)
    




            