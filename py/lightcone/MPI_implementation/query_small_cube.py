import numpy as np 
import matplotlib.pyplot as plt

path2disp = '/pscratch/sd/m/malvarez/websky-displacements/'
deltafile = '/global/cfs/cdirs/sobs/www/users/websky/ICs/Fvec_7700Mpc_n6144_nb30_nt16'

grid_max = 6144
subgrid_max = 128 

delta_small_cube = np.zeros((subgrid_max, subgrid_max, subgrid_max), dtype=np.float32)
sx_small_cube = np.zeros((subgrid_max, subgrid_max, subgrid_max), dtype=np.float32)
sy_small_cube = np.zeros((subgrid_max, subgrid_max, subgrid_max), dtype=np.float32)
sz_small_cube = np.zeros((subgrid_max, subgrid_max, subgrid_max), dtype=np.float32)

for k in range(subgrid_max):
    for j in range(subgrid_max):
        off = (k*grid_max**2 + j*grid_max) * 4 #in bytes
        # print(off)
        delta_small_cube[:,j,k] = np.fromfile(deltafile, count=subgrid_max, offset=off, dtype=np.float32)
        sx_small_cube[:,j,k] = np.fromfile(path2disp+'sx1_7700Mpc_n6144_nb30_nt16', count=subgrid_max, offset=off, dtype=np.float32)
        sy_small_cube[:,j,k] = np.fromfile(path2disp+'sy1_7700Mpc_n6144_nb30_nt16', count=subgrid_max, offset=off, dtype=np.float32)
        sz_small_cube[:,j,k] = np.fromfile(path2disp+'sz1_7700Mpc_n6144_nb30_nt16', count=subgrid_max, offset=off, dtype=np.float32)

plt.imshow(delta_small_cube[0,:,:], cmap=plt.cm.coolwarm, origin='lower')
plt.colorbar()
plt.show()