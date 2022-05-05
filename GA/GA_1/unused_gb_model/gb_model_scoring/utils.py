
from copy import deepcopy
from itertools import product
import pandas as pd
import numpy as np



def make_unit_list():
    '''
    Makes a list of dataframe of the SMILES of different types of building block units
    Contains: smi_filename, smiles, sTDDFTxtb_HOMO, calibrated B3LYP_HOMO, A_or_D, fused_ring_count

    Returns
    -------
    units: list
        list of dataframes containing SMILES, filenames, HOMO, A_or_D, and fused ring count
        [left_terminals, fused_cores, right_terminals]
    '''

    cores = pd.read_csv('../../../building_blocks/cores_with_ringcount.csv', index_col=0)
    left_terminals = pd.read_csv('../../../building_blocks/left_terminals_with_ringcount.csv', index_col=0)
    right_terminals = pd.read_csv('../../../building_blocks/right_terminals_with_ringcount.csv', index_col=0)

    # cores with 4 or more fused rings
    fused_cores = cores[cores['fused_ring_count'] > 3]
    # drops the old index
    fused_cores = fused_cores.reset_index(drop=True)

    # list of dataframes containing SMILES for terminal and fused cores
    units = [left_terminals, fused_cores, right_terminals]

    return units

def make_NFA_str(temp_NFA, unit_list):
    '''
    Makes the SMILES of the new molecule from the indices selected

    Parameters
    -----------
    temp_NFA: list
        list of indices in order of: [left terminal, core, right terminal]
    unit_list: list
        list of dataframes containing SMILES of each unit. [left_terminals, fused_cores, right_terminals]
    
    Returns
    ----------
    SMILES: str
        SMILES string of the new molecule
    '''

    left_term = temp_NFA[0]
    core = temp_NFA[1]
    right_term = temp_NFA[2]
    SMILES = unit_list[0].iloc[left_term][1] + unit_list[1].iloc[core][1] + unit_list[2].iloc[right_term][1]

    return SMILES

def make_filename(NFA):
    '''
    Makes a string of the indices for use as a file name

    Parameters
    ----------
    NFA: list
        list of indices: [term, core, term_R]

    Returns
    --------
    file_name: str
        filename in the formant: LT_C_RT 
        where LT is index of left terminal, C is index of core, and RT is indiex of right terminal
    '''

    # capture building block indexes as strings for file naming
    left_term = str(NFA[0])
    core = str(NFA[1])
    right_term = str(NFA[2])

    # make file name string for
    file_name = '%s_%s_%s' % (left_term, core, right_term)

    return file_name