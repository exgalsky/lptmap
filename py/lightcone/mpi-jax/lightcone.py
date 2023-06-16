import numpy as np 
import cosmology as cosmo 
import healpy as hp 
import liblightcone as llc
import backend as bk
import os

# ------ hardcoded parameters

grid_nside            = 768   # cube shape is parameterized by grid_nside; full resolution for websky is 6144
L_box                 = 7700. # periodic box size in comoving Mpc; lattice spacing is L_box / grid_nside
comov_lastscatter_Gpc = 13.8  # conformal distance to last scattering surface in Gpc
zmin                  = 0.05  # minimum redshift for projection (=0.05 for websky products)
zmax                  = 4.5   # maximum redshift for projection (=4.50 for websky products)
map_nside             = 1024

force_no_mpi          = False 
force_no_gpu          = False

print(f"LIGHTCONE: Setting backend...")
backend = bk.backend(force_no_mpi=force_no_mpi, force_no_gpu=force_no_gpu)
print(f"LIGHTCONE: Backend configuration complete.")

# Paths to displacement fields
try:
    path2disp = os.environ['LPT_DISPLACEMENTS_PATH']
except:
    path2disp = '/Users/shamik/Documents/Work/websky_datacube/'

print(f"LIGHTCONE: Path to displacement files set to {path2disp}")

sxfile = path2disp+'sx1_7700Mpc_n6144_nb30_nt16_no768'
syfile = path2disp+'sy1_7700Mpc_n6144_nb30_nt16_no768'
szfile = path2disp+'sz1_7700Mpc_n6144_nb30_nt16_no768'


print(f"LIGHTCONE: Computing cosmology...")
cosmo_wsp = cosmo.cosmology()
print(f"LIGHTCONE: Cosmology computed")

print(f"LIGHTCONE: Setting up lightcone workspace...")
lpt_wsp = llc.lightcone_workspace(cosmo_wsp, grid_nside, map_nside, L_box, zmin, zmax)

print(f"LIGHTCONE: Computing LPT to kappa map...")
kappa_map = lpt_wsp.lpt2map([sxfile, syfile, szfile], backend, bytes_per_cell=4)
print(f"LIGHTCONE: Kappa map computed. Saving to file.")


backend.mpi_backend.writemap2file(kappa_map, f'./output/kappa_map_grid-{ grid_nside }_nside-{ map_nside }.fits')
print(f"LIGHTCONE: Kappa map saved. Exiting...")




