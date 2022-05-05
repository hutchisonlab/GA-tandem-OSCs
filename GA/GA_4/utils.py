
from copy import deepcopy
from itertools import product
import pandas as pd
import numpy as np
from itertools import product


def make_unit_list():
    '''
    Makes a list of dataframe of the SMILES of different types of building block units
    Contains: smi_filename, smiles, sTDDFTxtb_HOMO, calibrated B3LYP_HOMO, A_or_D, fused_ring_count

    Returns
    -------
    units: list
        list of dataframes containing SMILES, filenames, HOMO, A_or_D, and fused ring count
        [left_terminals, fused_cores, right_terminals, spacers]
    '''

    units = pd.read_csv('GA4_donor_units.csv', index_col=0)

    return units


def make_file_name(polymer):
    '''
    Makes file name for a given polymer

    Parameters
    ---------
    polymer: list (specific format)
        [(#,#,...), A, B]

    Returns
    -------
    file_name: str
        polymer file name (w/o extension) showing monomer indicies and full sequence
        e.g. 100_200_101010 for a certain hexamer
    '''

    # capture monomer indexes as strings for file naming
    mono1 = str(polymer[1])
    mono2 = str(polymer[2])

    # make string of actual length sequence
    seq = list(polymer[0])
    seq = ''.join(map(str, seq))

    # make file name string
    file_name = '%s_%s_%s' % (mono1, mono2, seq)

    return file_name


def find_sequences(num_mono_species, poly_size):
    '''
    Finds all possible sequences

    Parameters
    ---------
    num_mono_species: int
        number of monomer species in each polymer (e.g. copolymer = 2)
    poly_size: int
        number of monomers per polymer

    Returns
    -------
    numer_seqs: list
        all possible sequences as a numerical list
    '''


    # find all possible sequences as numerical lists [start index 0]
    # (use cartesian product of each type of monomer over sequence length [poly_size])
    numer_seqs = list(product(range(num_mono_species), repeat=poly_size))

    return numer_seqs



def make_polymer_smi(temp_donor, unit_list):
    '''
    Constructs polymer string from monomers

    Parameters
    ---------
    polymer: list (specific format)
        [(#,#,...), A, B]
    smiles_list: list
        list of all possible monomer SMILES

    Returns
    -------
    poly_string: str
        polymer SMILES string
    '''
    donor_smiles = ''

    # cycle over monomer sequence until total number of monomers in polymer is reached
    for i in range(len(temp_donor[0])):
        # get monomer identifier from sequence
        seq_monomer_index = temp_donor[0][i]
        
        # find monomer index in smiles list and get monomer smiles
        monomer_index = temp_donor[seq_monomer_index + 1]
        donor_smiles = donor_smiles + unit_list.iloc[monomer_index][0]

    return donor_smiles
    
