
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
        [left_terminals, fused_cores, right_terminals, spacers]
    '''

    cores = pd.read_csv('../../building_blocks/cores_with_ringcount.csv', index_col=0)
    left_terminals = pd.read_csv('../../building_blocks/left_terminals_with_ringcount.csv', index_col=0)
    right_terminals = pd.read_csv('../../building_blocks/right_terminals_with_ringcount.csv', index_col=0)

    # cores with 4 or more fused rings
    fused_cores = cores[cores['fused_ring_count'] > 3]
    # drops the old index
    fused_cores = fused_cores.reset_index(drop=True)

    # core units with 3 or less rings to act as pi-conjugated spacers between fused core and terminal units
    spacers = cores[cores['fused_ring_count'] < 4]
    spacers = spacers.reset_index(drop=True)

    # list of dataframes containing SMILES for terminal and fused cores
    units = [left_terminals, fused_cores, right_terminals, spacers]

    return units

def make_NFA_str(temp_NFA, unit_list):
    '''
    Makes the SMILES of the new molecule from the indices selected

    Parameters
    -----------
    temp_NFA: list
        list of indices in order of: [left terminal, left spacer, core, right spacer, right terminal]
    unit_list: list
        list of dataframes containing SMILES of each unit. [left_terminals, fused_cores, right_terminals, spacers]
    
    Returns
    ----------
    SMILES: str
        SMILES string of the new molecule
    '''

    left_term = temp_NFA[0]
    left_spacer = temp_NFA[1]
    core = temp_NFA[2]
    right_spacer = temp_NFA[3]
    right_term = temp_NFA[4]
    SMILES = unit_list[0].iloc[left_term][1] +unit_list[3].iloc[left_spacer][1]+ unit_list[1].iloc[core][1] + unit_list[3].iloc[right_spacer][1] + unit_list[2].iloc[right_term][1]

    return SMILES

def make_filename(NFA):
    '''
    Makes a string of the indices for use as a file name

    Parameters
    ----------
    NFA: list
        list of indices: [term, left_spacer, core,right,spacer term_R]

    Returns
    --------
    file_name: str
        filename in the formant: LT_LS_C_RS_RT 
        where LT is index of left terminal, LS is index of left spacer, C is index of core, RS is index of right spacer, and RT is index of right terminal
    '''

    # capture building block indexes as strings for file naming
    left_term = str(NFA[0])
    left_spacer = str(NFA[1])
    core = str(NFA[2])
    right_spacer = str(NFA[3])
    right_term = str(NFA[4])

    # make file name string for
    file_name = '%s_%s_%s_%s_%s' % (left_term, left_spacer, core, right_spacer, right_term)

    return file_name

def find_sequence(NFA):
    '''
    Determines the symmetry/sequence of the molecule
    Parameters
    ----------
    NFA: list
        list of indices: [term, left_spacer, core, right_spacer, term_R]

    Returns
    --------
    sequence: int
        0 = symmetrical sequence (same end groups and spacers)
        1 = different end groups
        2 = different spacers
        3 = different spacers and end groups
    '''

    left_term = NFA[0]
    left_spacer = NFA[1]
    core = NFA[2]
    right_spacer = NFA[3]
    right_term = NFA[4]

    # symmetrical
    if left_term == right_term and left_spacer == right_spacer:
        sequence = 0
    # different end groups
    elif left_term != right_term and left_spacer == right_spacer:
        sequence = 1
    # different spacers
    elif left_term == right_term and left_spacer != right_spacer:
        sequence = 2
    # different spacers AND end groups
    else:
        sequence = 3

    return sequence

    
