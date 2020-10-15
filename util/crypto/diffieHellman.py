import nacl.bindings as nb
import random
import pandas as pd
import numpy as np
import math


def dict_keygeneration(peer_list):
  # CDB: turned these into dictionaries to relax assumptions around agent IDs.
  pkeys = {}
  skeys = {}

  for peer_id in peer_list:
    pkeys[peer_id], skeys[peer_id] = nb.crypto_kx_keypair()

  return pkeys, skeys


def dict_keyexchange(peer_list, self_id, my_pkeys, my_skeys, peer_pkeys):
  # CDB: The last three parameters are now all dictionaries.  Dictionary keys
  #      are peer ids to which we gave the key, or from which we received the key.
  #      comkeys is also now a dictionary keyed by peer id.
  comkeys = {}

  for peer_id in peer_list:
    if peer_id > self_id:
      common_key_raw, _ = nb.crypto_kx_client_session_keys(my_pkeys[peer_id], my_skeys[peer_id], peer_pkeys[peer_id])
    else:
      _, common_key_raw = nb.crypto_kx_server_session_keys(my_pkeys[peer_id], my_skeys[peer_id], peer_pkeys[peer_id])

    # Hash the common keys.
    comkeys[peer_id] = int.from_bytes(nb.crypto_hash_sha256(common_key_raw), byteorder='big')

  return comkeys


#PRG

def randomize( r, modulo, clientsign):
        # Call the double lenght pseudorsndom generator
        random.seed(r)
        rand            = random.getrandbits(256*2)
        rand_b_raw      = bin(rand)
        nr_zeros_append = 256 - (len(rand_b_raw) - 2)
        rand_b          = '0' * nr_zeros_append + rand_b_raw[2:]
        # Use first half to mask the inputs and second half as the next seed to the pseudorsndom generator
        R = int(rand_b[0:256], 2)
        r = int(rand_b[256:] , 2)
        return r, R 


def randomize_all(party_i, common_key_list, modulo):
    
    for i in range(len(common_key_list)):
        if i == party_i:
             continue
        clientsign = 1 if i > party_i else -1
        common_key_list[i], client = randomize( common_key_list[i], modulo, clientsign)
        
    return common_key_list, client
