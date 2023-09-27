import jax.numpy as jnp 
import backend as bk
import jax.random as rnd
import jax


import logging
log = logging.getLogger("LIGHTCONE")

force_no_mpi          = False 
force_no_gpu          = False
seed                  = 123456789
grid_size             = 768

backend = bk.backend(force_no_mpi=force_no_mpi, force_no_gpu=force_no_gpu)

data_shape = (grid_size, grid_size, grid_size)

start_mpi, stop_mpi = backend.mpi_backend.divide4mpi(data_shape,decom_type='slab', divide_axis=0)

key = rnd.PRNGKey(seed)

xidx_local = jnp.arange(start_mpi, stop_mpi)
yidx_local = jnp.arange(0, grid_size)
zidx_local = jnp.arange(0, grid_size)

@jax.jit
def idx2hash(x,y,z):
    return (( x*grid_size + y ) * grid_size + z).astype(jnp.int32)

@jax.jit
def uniq_normal_float32(key_in, hash_val):
    # hash_val = idx2hash(x, y, z)
    keys = rnd.fold_in(key_in, hash_val)
    return rnd.normal(keys, dtype=jnp.float32)


x_grid, y_grid, z_grid = jnp.meshgrid(xidx_local, yidx_local, zidx_local, indexing='ij')
hash_arr = jax.vmap(idx2hash, in_axes=(0,0,0), out_axes=0)(x_grid, y_grid, z_grid)
del x_grid, y_grid, z_grid

print(hash_arr.shape)

random_local = jax.vmap(uniq_normal_float32, in_axes=(None, 0), out_axes=0)(key, hash_arr.flatten())
random_local.reshape(hash_arr.shape)






