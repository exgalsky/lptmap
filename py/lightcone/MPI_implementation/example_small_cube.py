import numpy as np 
import matplotlib.pyplot as plt

# This is required only for generating a small 128 cube for a toy problem example

path2disp = '/pscratch/sd/m/malvarez/websky-displacements/'
deltafile = '/global/cfs/cdirs/sobs/www/users/websky/ICs/Fvec_7700Mpc_n6144_nb30_nt16'
outfile   = '/global/cfs/cdirs/mp107/exgal/data/xgsm/small_cube_128.npz'
grid_max = 6144
subgrid_max = 128 

delta = np.zeros((subgrid_max, subgrid_max, subgrid_max), dtype=np.float32)
sx = np.zeros((subgrid_max, subgrid_max, subgrid_max), dtype=np.float32)
sy = np.zeros((subgrid_max, subgrid_max, subgrid_max), dtype=np.float32)
sz = np.zeros((subgrid_max, subgrid_max, subgrid_max), dtype=np.float32)

for k in range(subgrid_max):
    print(k,range(subgrid_max))
    for j in range(subgrid_max):
        off = (k*grid_max**2 + j*grid_max) * 4 #in bytes
        # print(off)
        delta[:,j,k] = np.fromfile(deltafile, count=subgrid_max, offset=off, dtype=np.float32)
        sx[:,j,k] = np.fromfile(path2disp+'sx1_7700Mpc_n6144_nb30_nt16', count=subgrid_max, offset=off, dtype=np.float32)
        sy[:,j,k] = np.fromfile(path2disp+'sy1_7700Mpc_n6144_nb30_nt16', count=subgrid_max, offset=off, dtype=np.float32)
        sz[:,j,k] = np.fromfile(path2disp+'sz1_7700Mpc_n6144_nb30_nt16', count=subgrid_max, offset=off, dtype=np.float32)

plt.imshow(delta[0,:,:], cmap=plt.cm.coolwarm, origin='lower')
plt.colorbar()
plt.show()

np.savez(outfile,delta=delta,sx=sx,sy=sy,sz=sz)
