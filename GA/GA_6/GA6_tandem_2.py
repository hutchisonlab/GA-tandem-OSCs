# Imports
import csv
import os
import subprocess
import random
import numpy as np
from statistics import mean
from copy import deepcopy
import pickle


import utils
import scoring


def main():
    # flag for restart from save
    restart = 'n'
    # number of polymers in population
    pop_size = 32

    # Create list of possible acceptor and donor SMILES in specific format [acceptors, donors]
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


    for x in range(499):
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

    pop_size = params[0]
    unit_list= params[1]
    gen_counter= params[3]
    fitness_list= params[4] #[ranked_pairs, ranked_cmbined_abs_values, ranked_overlap_values, ranked_scores]

    gen_counter +=1

    ranked_population = fitness_list[0]

    # Selection - select heaviest (best) 50% of polymers as parents
    parent_population = parent_select(ranked_population)

    # Crossover & Mutation - create children to repopulate bottom 50% of NFAs in population
    new_population = crossover_mutate(parent_population, pop_size, unit_list)

    fitness_list = scoring.fitness_prediction(new_population, unit_list) #[ranked_pairs, ranked_cmbined_abs_values, ranked_overlap_values, ranked_scores]

    min_abs = fitness_list[1][-1]
    med_abs = fitness_list[1][int((len(fitness_list[1])-1)/2)]
    max_abs = fitness_list[1][0]

    min_overlap = fitness_list[2][-1]
    med_overlap = fitness_list[2][int((len(fitness_list[2])-1)/2)]
    max_overlap = fitness_list[2][0]

    min_score = fitness_list[-1][-1]
    med_score = fitness_list[-1][int((len(fitness_list[-1])-1)/2)]
    max_score = fitness_list[-1][0]


    with open('quick_analysis_data_GA6.csv', 'a') as quick_file:
        # write to quick analysis file
        quick_writer = csv.writer(quick_file)
        quick_writer.writerow([gen_counter, max_abs, min_abs, med_abs, min_overlap, max_overlap, med_overlap, min_score, med_score, max_score])

    for x in range(len(fitness_list[0])):
            pair = fitness_list[0][x]
            combined_abs = fitness_list[1][x]
            overlap = fitness_list[2][x]
            score = fitness_list[-1][x]

            with open('full_analysis_data_GA6.csv', 'a') as analysis_file:
                full_writer = csv.writer(analysis_file)
                full_writer.writerow([gen_counter, pair, combined_abs, overlap, score])

    params = [pop_size, unit_list, new_population, gen_counter, fitness_list]
    
    return(params)




def crossover_mutate(parent_population, pop_size, unit_list):
    new_pop = deepcopy(parent_population)

    # initialize new population with parents

    # loop until enough children have been added to reach population size
    while len(new_pop) < pop_size:

        # randomly select two parents (as indexes from parent list) to cross
        parent_a = random.randint(0, len(parent_population) - 1)
        parent_b = random.randint(0, len(parent_population) - 1)

        # ensure parents are unique indiviudals
        while parent_b == parent_a:
            parent_b = random.randint(0, len(parent_population) - 1)

        # create hybrid children
        temp_pair = []

        temp_acceptor = parent_population[parent_a][0]
        temp_donor = parent_population[parent_b][1]

        temp_pair.extend([temp_acceptor, temp_donor])

        # give child opportunity for mutation
        temp_pair = mutate(temp_pair, unit_list)

        acc_name = utils.find_name(temp_pair[0], "acc", unit_list)
        don_name = utils.find_name(temp_pair[1], "don", unit_list)

        acc_filename = '../GA1234/sTDDFT_xtb_output_GA123_acceptors/' + acc_name +'.stda'
        don_filename = '../GA_4/sTDDFTxtb_output/' + don_name + '.stda'

        acc_HOMO, acc_LUMO = utils.find_HOMO_LUMO(acc_filename)
        don_HOMO, don_LUMO = utils.find_HOMO_LUMO(don_filename)

        # checks to make sure acceptor frontier energies lower than donor's energy levels
        if acc_HOMO > don_HOMO:
            pass
        elif acc_LUMO > don_LUMO:
            pass
        else:
            if temp_pair in new_pop:
                pass
            else:
                new_pop.append(temp_pair)

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

    # new acceptor
    if random.randint(0, 1) == 0:
        temp_child[0] = random.randint(0, len(unit_list[0]) - 1)

        while temp_child[0] == 0:
            temp_child[0] = random.randint(0, len(unit_list[0]) - 1)

    # new donor
    else: 
        temp_child[1] = random.randint(0, len(unit_list[1]) - 1)

        while temp_child[1] == 2:
            temp_child[1] = random.randint(0, len(unit_list[1]) - 1)

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

    # create inital population as list of acceptor/donor pairs
    population = []

    while len(population) < pop_size:
        temp_pair = []

        acceptor = random.randint(0, len(unit_list[0]) - 1)
        # acceptor index 0 is the target acceptor for subcell 1
        while acceptor == 0:
            acceptor = random.randint(0, len(unit_list[0]) - 1)

        donor = random.randint(0, len(unit_list[1]) - 1)
        # donor index 0 is the target donor for subcell 1
        while donor == 2:
            donor = random.randint(0, len(unit_list[1]) - 1)

        acc_name = utils.find_name(acceptor, "acc", unit_list)
        don_name = utils.find_name(donor, "don", unit_list)

        acc_filename = '../GA1234/sTDDFT_xtb_output_GA123_acceptors/' + acc_name +'.stda'
        don_filename = '../GA_4/sTDDFTxtb_output/' + don_name + '.stda'

        acc_HOMO, acc_LUMO = utils.find_HOMO_LUMO(acc_filename)
        don_HOMO, don_LUMO = utils.find_HOMO_LUMO(don_filename)

        # checks to make sure acceptor frontier energies lower than donor's energy levels

        temp_pair.extend([acceptor, donor])

        if acc_HOMO > don_HOMO:
            pass
        elif acc_LUMO > don_LUMO:
            pass
        else:
            if temp_pair in population:
                pass
            else:
                population.append(temp_pair)

    # set up data directories
    setup = subprocess.call('mkdir predictions_GA6', shell=True)

    # create new analysis files
    with open('quick_analysis_data_GA6.csv', mode='w+') as quick:
        quick_writer = csv.writer(quick)
        quick_writer.writerow(['gen', 'max_abs', 'min_abs', 'med_abs', 'min_overlap', 'max_overlap', 'med_overlap', 'min_score', 'med_score', 'max_score'])

    with open('full_analysis_data_GA6.csv', mode='w+') as full:
        full_writer = csv.writer(full)
        full_writer.writerow(['gen', 'pair', 'combined_abs', 'overlap', 'score'])

    fitness_list = scoring.fitness_prediction(population, unit_list) #[ranked_pairs, ranked_cmbined_abs_values, ranked_overlap_values, ranked_scores]

    min_abs = fitness_list[1][-1]
    med_abs = fitness_list[1][int((len(fitness_list[1])-1)/2)]
    max_abs = fitness_list[1][0]

    min_overlap = fitness_list[2][-1]
    med_overlap = fitness_list[2][int((len(fitness_list[2])-1)/2)]
    max_overlap = fitness_list[2][0]

    min_score = fitness_list[-1][-1]
    med_score = fitness_list[-1][int((len(fitness_list[-1])-1)/2)]
    max_score = fitness_list[-1][0]


    with open('quick_analysis_data_GA6.csv', 'a') as quick_file:
        # write to quick analysis file
        quick_writer = csv.writer(quick_file)
        quick_writer.writerow([1, max_abs, min_abs, med_abs, min_overlap, max_overlap, med_overlap, min_score, med_score, max_score])

    for x in range(len(fitness_list[0])):
            pair = fitness_list[0][x]
            combined_abs = fitness_list[1][x]
            overlap = fitness_list[2][x]
            score = fitness_list[-1][x]

            with open('full_analysis_data_GA6.csv', 'a') as analysis_file:
                full_writer = csv.writer(analysis_file)
                full_writer.writerow([gen_counter, pair, combined_abs, overlap, score])

    params = [pop_size, unit_list, population, gen_counter, fitness_list]
    
    return(params)




if __name__ == '__main__':
    main()