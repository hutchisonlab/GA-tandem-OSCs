# Imports
import sklearn
print(sklearn.__version__)


import csv
import os
import subprocess
import random
import numpy as np
from scipy import stats
from statistics import mean
from copy import deepcopy
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem



import utils
import scoring

def main():
    # flag for restart from save
    restart = 'y'
    # number of polymers in population
    pop_size = 32

    # Create list of possible building block unit SMILES in specific format ([acc_term_L, don_core, acc_core, acc_term_R])
    unit_list = utils.make_unit_list()

    if restart == 'y':
        # reload parameters and random state from restart file
        open_params = open('last_gen.p', 'rb')
        params = pickle.load(open_params)
        open_params.close()

        open_rand = open('randstate.p', 'rb')
        randstate = pickle.load(open_rand)
        random.setstate(randstate)
        open_rand.close()

    else:
        # sets initial state
        randstate = random.getstate()
        rand_file = open('initial_randstate.p', 'wb')
        pickle.dump(randstate, rand_file)
        rand_file.close()

        # re-opens intial state during troubleshooting
        #open_rand = open('initial_randstate.p', 'rb')
        #randstate = pickle.load(open_rand)
        #random.setstate(randstate)
        #open_rand.close()

        # run initial generation if NOT loading from restart file
        params = init_gen(pop_size, unit_list)

        # pickle parameters needed for restart
        params_file = open('last_gen.p', 'wb')
        pickle.dump(params, params_file)
        params_file.close()

        # pickle random state for restart
        randstate = random.getstate()
        rand_file = open('randstate.p', 'wb')
        pickle.dump(randstate, rand_file)
        rand_file.close()


    for x in range(34):
        # run next generation of GA
        params = next_gen(params)

        # pickle parameters needed for restart
        params_file = open('last_gen.p', 'wb')
        pickle.dump(params, params_file)
        params_file.close()

        # pickle random state for restart
        randstate = random.getstate()
        rand_file = open('randstate.p', 'wb')
        pickle.dump(randstate, rand_file)
        rand_file.close()

def next_gen(params):
    # params = [pop_size, unit_list, population, gen_counter, fitness_list]

    
    pop_size = params[0]
    unit_list= params[1]
    gen_counter= params[3]
    fitness_list= params[4]

    gen_counter +=1

    ranked_population = fitness_list[3]

    # Selection - select heaviest (best) 50% of polymers as parents
    parent_population = parent_select(ranked_population)

    # Crossover & Mutation - create children to repopulate bottom 50% of NFAs in population
    new_population = crossover_mutate(parent_population, pop_size, unit_list)

    new_population_smiles = []
    for x in new_population:
        smiles = utils.make_NFA_str(x, unit_list)
        new_population_smiles.append(smiles)

    # run GFN2-xTB, sTD-DFT-xtb, and xtb solvation in hexane and water
    run_calculations(new_population_smiles, new_population)

    fitness_list = scoring.PCE_prediction(new_population) #[ranked_NFA_names, ranked_PCE, ranked_best_donor, ranked_population]

    min_PCE = fitness_list[1][-1]
    max_PCE = fitness_list[1][0]
    median = int((len(fitness_list[1])-1)/2)
    med_PCE = fitness_list[1][median]

    with open('quick_analysis_data_ensemble_rf_ann.csv', 'a') as quick_file:
        # write to quick analysis file
        quick_writer = csv.writer(quick_file)
        quick_writer.writerow([gen_counter, max_PCE, min_PCE, med_PCE])

    for x in range(len(fitness_list[0])):
        filename = fitness_list[0][x]
        donor = fitness_list[2][x]
        PCE = fitness_list[1][x]

        with open('full_analysis_data_ensemble_rf_ann.csv', 'a') as analysis_file:
            analysis_file.write('%d,%s,%f,%s,\n' % (gen_counter, filename, PCE, donor))

    
    params = [pop_size, unit_list, new_population, gen_counter, fitness_list]
    
    return(params)




def crossover_mutate(parent_population, pop_size, unit_list):
    new_pop = deepcopy(parent_population)
    new_pop_smiles = []

    # initialize new population with parents
    for parent in new_pop:
        parent_smiles = utils.make_NFA_str(parent, unit_list)
        new_pop_smiles.append(parent_smiles)


    # loop until enough children have been added to reach population size
    while len(new_pop) < pop_size:

        # randomly select two parents (as indexes from parent list) to cross
        parent_a = random.randint(0, len(parent_population) - 1)
        parent_b = random.randint(0, len(parent_population) - 1)

        # ensure parents are unique indiviudals
        if len(parent_population) > 1:
            while parent_b == parent_a:
                parent_b = random.randint(0, len(parent_population) - 1)

        parents = [parent_a, parent_b]

        # create hybrid children
        temp_child = []

        # left terminal unit
        chosen_parent = random.randint(0, 1)
        temp_child.append(parent_population[parents[chosen_parent]][0])

        # core unit
        chosen_parent = random.randint(0, 1)
        temp_child.append(parent_population[parents[chosen_parent]][1])

        # right terminal unit
        chosen_parent = random.randint(0, 1)
        temp_child.append(parent_population[parents[chosen_parent]][2])

        # give child opportunity for mutation
        temp_child = mutate(temp_child, unit_list)

        if temp_child in parent_population:
            pass
        else:
            new_pop.append(temp_child)

    return new_pop


def mutate(temp_child, unit_list):
    # set mutation rate
    mut_rate = 0.4

    # determine whether to mutate based on mutation rate
    rand = random.randint(1, 10)
    if rand <= (mut_rate * 10):
        pass
    else:
        return temp_child

    seq_or_unit = random.randint(0, 1)

    # mutate sequence to opposite of what it currently is
    if seq_or_unit == 0:
        # symmetrical
        if temp_child[0] == temp_child[2]:
            # right terminal
            term_R = random.randint(0, len(unit_list[2]) - 1)

            while term_R == temp_child[0]:
                term_R = random.randint(0, len(unit_list[2]) - 1)

            temp_child[2] = term_R
        # asymmetrical
        else:
            terminal = random.randint(0, 1)
            # use left terminal on both sides
            if terminal == 0:
                left_term = temp_child[0]
                temp_child[2] = left_term
            else:
                right_term = temp_child[2]
                temp_child[1] = right_term

    # mutate units. Can select either left terminal, core, or right terminal
    else:
        unit = random.randint(0, 2)
        # mutate left terminal
        if unit == 0:
            temp_child[0] = random.randint(0, len(unit_list[0]) - 1)
        # mutate core
        elif unit == 1:
            temp_child[1] = random.randint(0, len(unit_list[1]) - 1)
        else:
            temp_child[2] = random.randint(0, len(unit_list[2]) - 1)
    
    return temp_child

def parent_select(ranked_population):
    # find number of parents (half of population)
    parent_count = int(len(ranked_population) / 2)

    parent_list = []
    for x in range(parent_count):
        parent_list.append(ranked_population[x])

    return parent_list

def init_gen(pop_size, unit_list):
    # initialize generation counter
    gen_counter = 1

    # create inital population as list of NFAs
    population = []
    population_str = []
    counter = 0
    while counter < pop_size:
        temp_NFA = []

        symmetry = random.randint(0, 1) 

        # left terminal
        term = random.randint(0, len(unit_list[0]) - 1)
        # core 
        core = random.randint(0, len(unit_list[1]) - 1)

        # symmetric
        if symmetry == 0:
            temp_NFA.extend([term, core, term])

        # asymmetric
        else:
            # right terminal
            term_R = random.randint(0, len(unit_list[2]) - 1)

            while term_R == term:
                term_R = random.randint(0, len(unit_list[2]) - 1)
        
            temp_NFA.extend([term, core, term_R])

        # make SMILES string of NFA
        temp_NFA_SMILES = utils.make_NFA_str(temp_NFA, unit_list)

        # checks molecule for errors RDKit would catch
        try:
            # convert to canonical SMILES to check for duplication
            canonical_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(temp_NFA_SMILES))
        except:
            # prevents molecules with incorrect valence, which canonical smiles will catch and throw error
            print(temp_NFA_SMILES)
            print('Incorrect valence, could not perform canonical smiles')
            pass

        # check for duplication
        if canonical_smiles in population_str:
            pass
        else:

            population.append(temp_NFA)
            population_str.append(temp_NFA_SMILES)
            counter += 1

    # set up data directories
    setup = subprocess.call('mkdir PCE_predictions_ensemble_rf_ann FF_optimized GFN2_output sTDDFTxtb_output solvation_hexane solvation_water', shell=True)
  
    # run GFN2-xTB, sTD-DFT-xtb, and xtb solvation in hexane and water
    run_calculations(population_str, population)

    # create new analysis files
    with open('quick_analysis_data_ensemble_rf_ann.csv', mode='w+') as quick:
        quick_writer = csv.writer(quick)
        quick_writer.writerow(['gen', 'min_PCE', 'max_PCE', 'med_PCE'])

    with open('full_analysis_data_ensemble_rf_ann.csv', mode='w+') as full:
        full_writer = csv.writer(full)
        full_writer.writerow(['gen', 'filename', 'PCE', 'donor'])

    fitness_list = scoring.PCE_prediction(population) #[ranked_NFA_names, ranked_PCE, ranked_best_donor, ranked_population]

    min_PCE = fitness_list[1][-1]
    max_PCE = fitness_list[1][0]
    median = int((len(fitness_list[1])-1)/2)
    med_PCE = fitness_list[1][median]

    with open('quick_analysis_data_ensemble_rf_ann.csv', 'a') as quick_file:
        # write to quick analysis file
        quick_writer = csv.writer(quick_file)
        quick_writer.writerow([1, max_PCE, min_PCE, med_PCE])

    for x in range(len(fitness_list[0])):
            filename = fitness_list[0][x]
            donor = fitness_list[2][x]
            PCE = fitness_list[1][x]

            with open('full_analysis_data_ensemble_rf_ann.csv', 'a') as analysis_file:
                analysis_file.write('%d,%s,%f,%s,\n' % (gen_counter, filename, PCE, donor))
    
    params = [pop_size, unit_list, population, gen_counter, fitness_list]
    
    return(params)


def run_calculations(population_str, population):
    '''
    Run all relevant calculations (FF optimization, GFN2-xTB, sTDDFT-xTB, and xtb solvation energies)
    
    Parameters
    ----------
    population_str: list
        list of SMILES of population
    population: list
        list of list of indices of NFAs
    '''

    for x in range(len(population_str)):
        filename = utils.make_filename(population[x])
        print(filename)

        # Check if this molecule has already been run in a previous generation
        duplicate = check_duplicate(filename)

        # if its a new molecule
        if duplicate == False:
            # force field geometry optimization
            ff_optimization(population_str[x], filename)
            
            # GFN2-xTB geoemtry optimization
            GFN2_optimization(filename)

            # sTD-DFT-xTB calculation
            sTDDFT_xTB(filename)

            # run xtb solvation calculations in water and hexane
            solvation(filename)


def solvation(filename):
    '''
    Calculate solvation energies in water and hexane with xtb

    Parameters
    ----------
        filename: str
    '''
    exists = os.path.isfile('../solvation_hexane/%s.out' % (filename))
    if exists:
        print("hexane output file existed")
    else:
        # make directory to run xtb in for the NFA
        mkdir_NFA = subprocess.call('(mkdir ../solvation_hexane/%s)' % (filename), shell=True)

        # run sTD-DFT-xTB calculation
        hexane = subprocess.call('(cd ../solvation_hexane/%s && /ihome/ghutchison/oda6/xtb/xtb-641/bin/xtb ../../GFN2_output/%s.mol --sp --alpb hexane > ../%s.out)' %(filename, filename, filename), shell=True)

        # delete xtb run directory for the NFA
        del_NFAdir = subprocess.call('(rm -r ../solvation_hexane/%s)' % (filename), shell=True)

    exists = os.path.isfile('../solvation_water/%s.out' % (filename))
    if exists:
        print("water output file existed")
    else:
        # make directory to run xtb in for the NFA
        mkdir_NFA = subprocess.call('(mkdir ../solvation_water/%s)' % (filename), shell=True)

        # run sTD-DFT-xTB calculation
        hexane = subprocess.call('(cd ../solvation_water/%s && /ihome/ghutchison/oda6/xtb/xtb-641/bin/xtb ../../GFN2_output/%s.mol --sp --alpb water > ../%s.out)' %(filename, filename, filename), shell=True)

        # delete xtb run directory for the NFA
        del_NFAdir = subprocess.call('(rm -r ../solvation_water/%s)' % (filename), shell=True)

def sTDDFT_xTB(filename):
    '''
    Perform sTD-DFT-xtb

    Parameters
    ----------
        filename: str
    '''
    exists = os.path.isfile('../sTDDFTxtb_output/%s.stda' % (filename))
    if exists:
        print("sTDDFTxtb output file existed")
    else:

        # convert optimized mol files to xyz
        mol_to_xyz = subprocess.call('(cd ../GFN2_output && obabel %s.mol -O %s.xyz)' %(filename, filename), shell=True)

        # make directory to run xtb in for the NFA
        mkdir_NFA = subprocess.call('(mkdir ../sTDDFTxtb_output/%s)' % (filename), shell=True)

        # run sTD-DFT-xTB calculation
        #stddftxtb = subprocess.call('(export XTB4STDAHOME=/ihome/ghutchison/blp62/xtb4stda && export PATH=$PATH:${XTB4STDAHOME}/bin && cd sTDDFTxtb_output/%s && ${XTB4STDAHOME}/bin/xtb4stda ../../GFN2_output/%s.xyz > ../%s.out && ${XTB4STDAHOME}/bin/stda -xtb -e 5 -rpa > ../%s.stda)' %(filename, filename, filename, filename), shell=True)
        stddftxtb = subprocess.call('(export XTB4STDAHOME=/ihome/ghutchison/blp62/xtb4stda && cd ../sTDDFTxtb_output/%s && /ihome/ghutchison/blp62/xtb4stda/bin/xtb4stda ../../GFN2_output/%s.xyz > ../%s.out && /ihome/ghutchison/blp62/xtb4stda/bin/stda -xtb -e 5 -rpa > ../%s.stda)' %(filename, filename, filename, filename), shell=True)


        # delete xtb run directory for the NFA
        del_NFAdir = subprocess.call('(rm -r ../sTDDFTxtb_output/%s)' % (filename), shell=True)

def GFN2_optimization(filename):
    '''
    Optimize geometry with GFN2-xTB

    Parameters
    ----------
        filename: str
    '''
    exists = os.path.isfile('../GFN2_output/%s.mol' % (filename))
    if exists:
        print("GFN2 output file existed")
    else:
        # make directory to run xtb in for the NFA
        mkdir_NFA = subprocess.call('(mkdir ../GFN2_output/%s)' % (filename), shell=True)

        # run xTB geometry optimization
        xtb = subprocess.call('(cd ../GFN2_output/%s && /ihome/ghutchison/oda6/xtb/xtb-641/bin/xtb ../../FF_optimized/%s.mol -opt > ../%s.out)' %(filename, filename, filename), shell=True)

        # save optimized mol file
        save_opt_file = subprocess.call('(cp ../GFN2_output/%s/xtbopt.mol ../GFN2_output/%s.mol)' % (filename, filename), shell=True)

        # delete xtb run directory for the NFA
        del_NFAdir = subprocess.call('(rm -r ../GFN2_output/%s)' % (filename), shell=True)


def check_duplicate(filename):
    '''
    Check to see if this molecule was already run in a previous generation

    Parameters
    ----------
        filename: str
    '''
    # if output file already exists, skip calculations
    exists = os.path.isfile('../sTDDFTxtb_output/%s.out' % (filename))
    if exists:
        print("output file existed")
        return True
    else:
        return False

def ff_optimization(smiles, filename):
    '''
    Generates 3D mol file with force field optimization from SMILES

    Parameters
    ----------
    smiles: str
        SMILES of the molecule
    filename: str
        string of the indices to be used for filenames: LT_C_RT 
    
    '''

    exists = os.path.isfile('../FF_optimized/%s.mol' % (filename))
    if exists:
        print("FF optimized mol file existed")
    else:
        # create mol object from SMILES
        mol = Chem.MolFromSmiles(smiles)
        # add hydrogens
        m2 = Chem.AddHs(mol)
        
        AllChem.EmbedMolecule(m2,randomSeed=0xf00d, useRandomCoords=True)

        # Se does not perform well with UFF. Need to use MMFF94
        if '[se]' in smiles:
            AllChem.MMFFOptimizeMolecule(m2)
        else:
            AllChem.UFFOptimizeMolecule(m2)
        
        new_filename = '../FF_optimized/' + filename + '.mol'

        print(Chem.MolToMolBlock(m2),file=open(new_filename,'w+'))


if __name__ == '__main__':
    main()