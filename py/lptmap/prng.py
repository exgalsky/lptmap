import numpy as np
import jax
import jax.numpy as jnp 
import jax.random as rdm 
import healpy as hp
from functools import partial

seed = 57885161         # Mersenne exponents no 48. Ref: https://oeis.org/A000043
nseq = 512

# Do we hard code this here, or have it defined elsewhere and read here?
cmb_sequence = [[0, 'tlm'], [1, 'elm'], [2, 'blm'], [3, 'dlm']]
dust_sequence = [[4, 'dust_tlm'], [5, 'dust_elm'], [6, 'dust_blm']]
sync_sequence = [[7, 'sync_tlm'], [8, 'sync_elm'], [9, 'sync_blm']]
ic_sequence = [[32, 'ic']]

master_key  = rdm.PRNGKey(seed)
key_sequence = rdm.split(master_key, num=nseq)
sequence_no = np.arange(0, nseq, dtype=np.int16)
sequence_id = [f"{seed}-{seq_no:03}" for seq_no in sequence_no]
registry    = [None for _ in sequence_no]

lmax = 3 * 8192 - 1
max_harmonic_size = int(((lmax+1)*(lmax+2))/2)

for cmb_stream in cmb_sequence:
    registry[cmb_stream[0]] = cmb_stream[1] 

for dust_stream in dust_sequence:
    registry[dust_stream[0]] = dust_stream[1] 

for sync_stream in sync_sequence:
    registry[sync_stream[0]] = sync_stream[1] 

def register_component(component_name, component_type):
    unassigned_idx = [i for i, val in enumerate(registry) if val == None]
    unassigned_stream = sequence_no[unassigned_idx]

    if component_type.lower() == 'galactic':
        registry[np.min(unassigned_stream[unassigned_stream >= 4 & unassigned_stream < 64])] = component_name.lower()
    elif component_type.lower() == 'extragalactic':
        registry[np.min(unassigned_stream[unassigned_stream >= 64 & unassigned_stream < 128])] = component_name.lower()
    else:
        registry[np.min(unassigned_stream[unassigned_stream >= 128])] = component_name.lower()

def check4component(component_name):
    if component_name in registry:
        return sequence_no[registry.index(component_name.lower())]
    else:
        return None
    
def __normal(key, data_ids):

    @jax.jit
    def __fetch_normal(key, data_id):
        key_new = rdm.fold_in(key, data_id)
        return rdm.normal(key_new)
    

    
class PRNG_stream:
    def __init__(self, component_name, component_type='', harmonic=False):
        
        component_no = check4component(component_name)
        if component_no == None : register_component(component_name, component_type)
        self.component_no = check4component(component_name)
        self.component_name = registry[self.component_no]
        self.component_id = sequence_id[self.component_no]
        self.component_key = key_sequence[self.component_no]
        self.harmonic = harmonic
        
    def normal()