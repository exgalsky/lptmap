| Line No. |   PCM (in bytes) increment   |   PCM (in bytes) total  |   Total memory (MB)  |   Increment (MB)  |  Code |
|   :---:  |    :----:                    |     :----:              |     :---:            |     :---:         | :---  |
|  73  |    -    |   -  |    96   |    96     |   `skymap = np.zeros((npix,))`  |
|  106 |   4   |   4   |     1728  |   1824   |   `grid_sx = read_displacement(sxfile)` |
|  107 |   4   |   8   |     1728  |   3552   |   `grid_sy = read_displacement(syfile)` |
|  108 |   4   |   12   |    1728  |   5280   |   `grid_sz = read_displacement(szfile)` |
|  117 |   12  |   24   |    5184  |   10464  |   `grid_qx, grid_qy, grid_qz = lagrange_mesh(xaxis, yaxis, zaxis, translation, lattice_size_in_Mpc)`|
|  121 |  4   |   28   |   1728   |   12192  |  `lagrange_grid = jax.vmap(comoving_q, in_axes=(0, 0, 0, None, None), out_axes=0)(grid_qx, grid_qy, grid_qz, translation, lattice_size_in_Mpc)` |
|  124 |  4   | 32  |  1728  |  13920  |  `redshift_grid = jax.vmap(cosmo_wsp.comoving_distance2z)(lagrange_grid)` |
|  128 | 8 |   40  |  3456  |  17376 |  `ipix_grid = jhp.vec2pix(nside, grid_qz, grid_qy, grid_qx)` |
| 132 |  4 |   44  |  1728  |  19104 |   `kernel_sphere = jnp.where((lagrange_grid >= chimin) & (lagrange_grid <= chimax), jax.vmap(lensing_kernel_F)(lagrange_grid, redshift_grid), 0.)` |
| 137 | -12 |  32  |  -5184  |  13920 |  `del kernel_sphere, ipix_grid` |
| 141 | 4  |  36  |  1728  |  15648  |   `jax.vmap(cosmo_wsp.growth_factor_D)(redshift_grid)` |
| 146 | 4  |  40 |  1728 |  17376 | `grid_Xx = jax.vmap(euclid_i, in_axes=(0, 0, 0, None, None), out_axes=0)(grid_qx, grid_sx, growth_grid, lattice_size_in_Mpc, translation[0])`|
| 147 | -4  |  36 |  -1728 |  15648 | `del grid_qx` |
| 151 | 4 | 40 | 1728 | 17376 |  `grid_Xy = jax.vmap(euclid_i, in_axes=(0, 0, 0, None, None), out_axes=0)(grid_qy, grid_sy, growth_grid, lattice_size_in_Mpc, translation[1])`|
| 142 | -4  |  36 |  -1728 | 15648 | `del grid_qy` |
| 156 | 4 | 40 | 1728 | 17376 | `grid_Xz = jax.vmap(euclid_i, in_axes=(0, 0, 0, None, None), out_axes=0)(grid_qz, grid_sz, growth_grid, lattice_size_in_Mpc, translation[2])` |
| 157 | -8 | 32 | -3456 | 13920 | `del grid_qz, growth_grid` |
| 163 | 8 | 40 | 3456 | 17376 | `ipix_grid = jhp.vec2pix(nside, grid_Xz, grid_Xy, grid_Xx)` |
| 164 | -12 | 28 | -5184 | 12192 | `del grid_Xx, grid_Xy, grid_Xz ` |
| 168 | 4 | 32 | 1728 | 13920 | `kernel_sphere = jnp.where((lagrange_grid >= chimin) & (lagrange_grid <= chimax), jax.vmap(lensing_kernel_F)(lagrange_grid, redshift_grid), 0.) ` |
| 168 | -8 | 24 | -3456 | 10464 | `del lagrange_grid, redshift_grid` |
| 174 | -12 | 12 | -5184 | 5184 |  `del ipix_grid, kernel_sphere` |
| 178 | -12 | 12 | -5184 | 96 | `if store_displacements: del grid_sx, grid_sy, grid_sz` |
