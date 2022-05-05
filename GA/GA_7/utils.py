
from copy import deepcopy
from itertools import product
import pandas as pd
import numpy as np


def make_unit_list():
    '''
    Makes a list of dataframes of top 1000 acceptor and donor names from GAs 1-4

    Returns
    -------
    units: list
        list of dataframes containing acceptor and donor names
        [acceptors, donors]
    '''

    acceptors = pd.read_csv('../GA1234/top_acceptors_GA1234.csv')
    donors = pd.read_csv('../GA1234/top_donors_GA1234.csv')

    units = [acceptors, donors]

    return units


def make_filename(pair):
    '''
    Makes a string of the indices for use as a file name

    Parameters
    ----------
    pair: list
        index of acceptor and donor

    Returns
    --------
    file_name: str
        filename in the formant: LT_LS_C_RS_RT 
        where LT is index of left terminal, LS is index of left spacer, C is index of core, RS is index of right spacer, and RT is index of right terminal
    '''

    # capture building block indexes as strings for file naming
    acceptor = str(pair[0])
    donor = str(pair[1])

    # make file name string for
    file_name = '%s_%s' % (acceptor, donor)

    return file_name

def find_name(index, mol_type, unit_list):
    if mol_type == 'acc':
        return unit_list[0].iloc[index][0]
    elif mol_type == 'don':
        return unit_list[1].iloc[index][0]

def find_HOMO_LUMO(filename):   
    with open(filename, 'r', encoding = 'utf-8') as file:
        line = file.readline()
        while line:
            if 'ordered frontier orbitals' in line:
                for x in range(11):
                    line = file.readline()
                
                line = file.readline()
                line_list = line.split()
                HOMO = float(line_list[1])
                
                line = file.readline()
                line = file.readline()
                line_list = line.split()
                LUMO = float(line_list[1])

            line = file.readline()  
        line = file.readline()

    return HOMO, LUMO
