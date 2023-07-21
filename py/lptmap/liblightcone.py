import numpy as np 
import healpy as hp
import jax_healpix as jhp
import kernel_lib as kl
from functools import partial
import jax
import jax.numpy as jnp
from time import time

import logging
log = logging.getLogger(__name__)

@partial(jax.jit, static_argnames=['trans_vec', 'Dgrid_in_Mpc'])
def lagrange_mesh(x_axis, y_axis, z_axis, trans_vec, Dgrid_in_Mpc):
    qx, qy, qz = jnp.meshgrid( (x_axis + 0.5 + trans_vec[0]) * Dgrid_in_Mpc, (y_axis + 0.5 + trans_vec[1]) * Dgrid_in_Mpc, (z_axis + 0.5 + trans_vec[2]) * Dgrid_in_Mpc, indexing='ij')
    return qx.ravel(), qy.ravel(), qz.ravel()

@jax.jit
def comoving_q(x_i, y_i, z_i):
    return jnp.sqrt(x_i**2. + y_i**2. + z_i**2.)

@jax.jit
def euclid_i(q_i, s_i, growth_i):
    return (q_i + growth_i * s_i)

def _read_displacement(filename, chunk_shape, chunk_offset):
    return np.fromfile(filename, count=chunk_shape[0] * chunk_shape[1] * chunk_shape[2], offset=chunk_offset, dtype=np.float32)

def _profiletime(task_tag, step, times):
    dt = time() - times['t0']
    log.usky_debug(f'{task_tag}: {dt:.6f} sec for {step}')
    if step in times.keys():
        times[step] += dt
    else:
        times[step] = dt
    times['t0'] = time()
    return times

def _sortdict(dictin,reverse=False):
    return dict(sorted(dictin.items(), key=lambda item: item[1], reverse=reverse))

def _summarizetime(task_tag, times):
    total_time = 0
    for key in _sortdict(times,reverse=True).keys():
        if key != 't0':
            log.usky_info(f'{task_tag}: {times[key]:.5e} {key}')
            total_time += times[key]
    log.usky_info(f'{task_tag}: {total_time:.5e} all steps')

class lightcone_workspace():
    def __init__(self, cosmo_workspace, grid_nside, map_nside, box_length_in_Mpc, zmin, zmax):
        self.grid_nside = grid_nside 
        self.map_nside = map_nside
        self.npix = hp.nside2npix(self.map_nside)
        self.L_box = box_length_in_Mpc
        self.cosmo = cosmo_workspace
        self.chimin = self.cosmo.comoving_distance(zmin)
        self.chimax = self.cosmo.comoving_distance(zmax)
        
    def grid2map(self, sx, sy, sz, grid_xstarts, grid_xstops, grid_ystarts, grid_ystops, grid_zstarts, grid_zstops, backend=None):

        tgridmap0 = time()
        overalltimes = {}
        times = {}
        overalltimes={'t0' : time()}
        times={'t0' : time()}
        log = logging.getLogger(__name__)

        task_tag0 = ""
        if backend is not None:
            task_tag0 = backend.jax_backend.task_tag
        task_tag = task_tag0

        # Lattice spacing (a_latt in Websky parlance) in Mpc
        lattice_size_in_Mpc = self.L_box / self.grid_nside

        solidang_pix = 4*np.pi / self.npix

        # Effectively \Delta chi, comoving distance interval spacing for LoS integral
        geometric_factor = lattice_size_in_Mpc**3. / solidang_pix
        times = _profiletime(task_tag, 'initialization', times)

        # Setup axes for the slab grid
        xaxis = jnp.arange(grid_xstarts, grid_xstops, dtype=jnp.int16)
        yaxis = jnp.arange(grid_ystarts, grid_ystops, dtype=jnp.int16)
        zaxis = jnp.arange(grid_zstarts, grid_zstops, dtype=jnp.int16)
        times = _profiletime(task_tag, 'slab grid axis setup', times)

        skymap = jnp.zeros((self.npix,))
        times = _profiletime(task_tag, 'skymap init', times)

        shift_param = self.grid_nside
        origin_shift = [(0,0,0), (-shift_param,0,0), (0,-shift_param,0), (-shift_param,-shift_param,0),
                        (0,0,-shift_param), (-shift_param,0,-shift_param), (0,-shift_param,-shift_param), (-shift_param,-shift_param,-shift_param)]
        times = _profiletime(task_tag, 'origin shift', times)

        t0 = time()
        for translation in origin_shift:

            # Lagrangian coordinates
            qx, qy, qz = lagrange_mesh(xaxis, yaxis, zaxis, translation, lattice_size_in_Mpc)
            times = _profiletime(task_tag, 'Lagrangian meshgrid', times)

            # comoving distance
            chi = jax.vmap(comoving_q, in_axes=(0, 0, 0), out_axes=0)(qx, qy, qz)    # 4 : 22
            times = _profiletime(task_tag, 'chi', times)

            # redshift
            redshift = jax.vmap(self.cosmo.comoving_distance2z)(chi)
            times = _profiletime(task_tag, 'redshift', times)

            # healpix indices
            ipix = jhp.vec2pix(self.map_nside, qz, qy, qx)
            times = _profiletime(task_tag, 'ipix', times)

            # lensing kernel
            kernel = -jnp.where((chi >= self.chimin) & (chi <= self.chimax), jax.vmap(kl.lensing_kernel_F, in_axes=(None, None, 0, 0), out_axes=0 )(self.cosmo, geometric_factor, chi, redshift), 0.)
            times = _profiletime(task_tag, 'kernel', times)

            # add lensing kernel to corresponding skymap pixel at each grid position
            skymap = skymap.at[ipix].add(kernel)
            times = _profiletime(task_tag, 'skymap add', times)

            del kernel, ipix
            times = _profiletime(task_tag, 'delete kernel, ipix', times)

            # linear growth factor
            growth = jax.vmap(self.cosmo.growth_factor_D)(redshift)
            times = _profiletime(task_tag, 'growth', times)

            # Eulerian x coordinate
            Xx = jax.vmap(euclid_i, in_axes=(0, 0, 0), out_axes=0)(qx, sx, growth)
            times = _profiletime(task_tag, 'Xx', times)

            del qx
            times = _profiletime(task_tag, 'qx delete', times)

            # Eulerian y coordinate
            Xy = jax.vmap(euclid_i, in_axes=(0, 0, 0), out_axes=0)(qy, sy, growth)
            times = _profiletime(task_tag, 'Xy', times)

            del qy
            times = _profiletime(task_tag, 'qy delete', times)

            # Eulerian z coordinate
            Xz = jax.vmap(euclid_i, in_axes=(0, 0, 0), out_axes=0)(qz, sz, growth)
            times = _profiletime(task_tag, 'Xz', times)

            del qz, growth
            times = _profiletime(task_tag, 'qz, growth delete', times)

            ipix = jhp.vec2pix(self.map_nside, Xz, Xy, Xx)
            times = _profiletime(task_tag, 'ipix Eulerian', times)

            del Xx, Xy, Xz
            times = _profiletime(task_tag, 'Xx, Xy, Xz delete', times)

            kernel = jnp.where((chi >= self.chimin) & (chi <= self.chimax),
                               jax.vmap(kl.lensing_kernel_F, in_axes=(None, None, 0, 0), out_axes=0 )
                                       (self.cosmo, geometric_factor, chi, redshift), 0.)
            times = _profiletime(task_tag, 'kernel Eulerian', times)

            del chi, redshift
            times = _profiletime(task_tag, 'chi, redshift delete Eulerian', times)

            skymap = skymap.at[ipix].add(kernel)
            times = _profiletime(task_tag, 'skymap add Eulerian', times)

            del ipix, kernel
            times = _profiletime(task_tag, 'ipix, kernel delete Eulerian', times)

        del sx, sy, sz
        times = _profiletime(task_tag+' (grid2map)', 'sx, sy, sz delete', times)
        _summarizetime(task_tag+' (grid2map steps)',times)

        overalltimes = _profiletime(task_tag, 'grid2map', overalltimes)
        _summarizetime(task_tag+' (grid2map)',overalltimes)

        return skymap
    
    def lpt2map(self, dispfilenames, backend, bytes_per_cell=4, use_tqdm=False):        #kernel_list,

        data_shape = (self.grid_nside, self.grid_nside, self.grid_nside)
        # HARDCODED PARAMETERS -- NEED TO DOCUMENT AND IMPLEMENT USER SETTING AT RUNTIME
        peak_per_cell_memory = 75.0
        jax_overhead_factor  = 1.5
        backend.datastream_setup(data_shape, bytes_per_cell, peak_per_cell_memory, jax_overhead_factor, decom_type='slab', divide_axis=0)
        jax_iterator = backend.get_iterator()
        obs_map = np.zeros((self.npix,))
        task_tag = backend.jax_backend.task_tag
        iterator = jax_iterator
        if use_tqdm:
            from tqdm import tqdm
            iterator = tqdm(jax_iterator, ncols=120)
        else:
            i=0 ; t=0. ; tbar=0. ; tread=0. ; tmap=0.
            n=len(iterator)
        for iter in iterator:

            log.usky_debug(f"start, stop, offset, shape: { iter }", per_task=True)

            if not use_tqdm:
                t1=time()

            sx = _read_displacement(dispfilenames[0], iter[3], iter[2])
            sy = _read_displacement(dispfilenames[1], iter[3], iter[2])
            sz = _read_displacement(dispfilenames[2], iter[3], iter[2])

            if not use_tqdm:
                t2=time()

            times={'t0' : time()}

            sx = jnp.asarray(sx) ; sy = jnp.asarray(sy) ; sz = jnp.asarray(sz)
            times = _profiletime(task_tag, 'numpy to jax sx, sy, sz', times)

            obs_map_cur = self.grid2map(sx, sy, sz, iter[0], iter[1], 0, self.grid_nside, 0, self.grid_nside, backend=backend)
            times = _profiletime(task_tag, 'grid2map in lpt2map', times)

            obs_map_cur = np.array(obs_map_cur, dtype=np.float32)
            times = _profiletime(task_tag, 'jax to numpy obs_map', times)

            obs_map += obs_map_cur  #, kernel_list
            times = _profiletime(task_tag, 'accumulate obs_map', times)

            _summarizetime(task_tag+' (lpt2map mapmaking)', times)

            if not use_tqdm:
                t3=time()
                i += 1
                dtread = t2-t1
                dtmap  = t3-t2
                tread += dtread
                tmap  += dtmap
                tread_bar = tread / i
                tmap_bar  = tmap  / i
                log.usky_info(f"{task_tag}: for iteration {i}/{n}: IO , mapping = {dtread:.3f} , {dtmap:.3f} (mean = {tread_bar:.3f} , {tmap_bar:.3f}; " +
                              f"total = {tread:.3f} , {tmap:.3f})")

        return backend.mpi_backend.reduce2map(obs_map)
    




            