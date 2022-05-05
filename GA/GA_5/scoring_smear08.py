import numpy as np
import pandas as pd
from pickle import load
import os

import utils


def calc_combined_abs(subcell1_acc_spectraIntensity, subcell1_don_spectraIntensity,  acc_spectraIntensity, don_spectraIntensity, don_spectraNM):
    
    combined_intensities = [(subcell1_acc_spectraIntensity[i] + subcell1_don_spectraIntensity[i] + acc_spectraIntensity[i] + don_spectraIntensity[i]) for i in range(len(don_spectraIntensity))]
    combined_abs_integral = np.trapz(np.flip(combined_intensities), np.flip(don_spectraNM), dx=0.1, axis=- 1)

    return combined_abs_integral


def calc_overlap(subcell1_acc_spectraIntensity, subcell1_don_spectraIntensity,  acc_spectraIntensity, don_spectraIntensity, don_spectraNM):
    
    # subcell 1 acceptor with subcell 1 donor
    subcell1_A_subcell1_D = [(subcell1_acc_spectraIntensity[i] * subcell1_don_spectraIntensity[i]) for i in range(len(subcell1_don_spectraIntensity))]
    # multiplies the intensities so if one is zero is cancels it out. Want to minimize the overlap
    overlap1 = np.trapz(np.flip(subcell1_A_subcell1_D), np.flip(don_spectraNM), dx=0.1, axis=- 1)

    # subcell 1 acceptor with subcell 2 acceptor
    subcell1_A_subcell2_A = [(subcell1_acc_spectraIntensity[i] * acc_spectraIntensity[i]) for i in range(len(acc_spectraIntensity))]
    overlap2 = np.trapz(np.flip(subcell1_A_subcell2_A), np.flip(don_spectraNM), dx=0.1, axis=- 1)

    # subcell 1 acceptor with subcell 2 donor
    subcell1_A_subcell2_D = [(subcell1_acc_spectraIntensity[i] * don_spectraIntensity[i]) for i in range(len(don_spectraIntensity))]
    overlap3 = np.trapz(np.flip(subcell1_A_subcell2_D), np.flip(don_spectraNM), dx=0.1, axis=- 1)

    # subcell 1 donor with subcell 2 acceptor
    subcell1_D_subcell2_A = [(subcell1_don_spectraIntensity[i] * acc_spectraIntensity[i]) for i in range(len(acc_spectraIntensity))]
    overlap4 = np.trapz(np.flip(subcell1_D_subcell2_A), np.flip(don_spectraNM), dx=0.1, axis=- 1)

    # subcell 1 donor with subcell 2 donor
    subcell1_D_subcell2_D = [(subcell1_don_spectraIntensity[i] * don_spectraIntensity[i]) for i in range(len(don_spectraIntensity))]
    overlap5 = np.trapz(np.flip(subcell1_D_subcell2_D), np.flip(don_spectraNM), dx=0.1, axis=- 1)

    # subcell 2 acceptor with subcell 2 donor
    subcell2_A_subcell2_D = [(acc_spectraIntensity[i] * don_spectraIntensity[i]) for i in range(len(don_spectraIntensity))]
    overlap6 = np.trapz(np.flip(subcell2_A_subcell2_D), np.flip(don_spectraNM), dx=0.1, axis=- 1)

    total_overlap = overlap1 + overlap2 + overlap3 + overlap4 + overlap5 + overlap6
        
    return total_overlap




def fitness_prediction(population, unit_list):
    """
    Calculates the PCE of the best acceptor-donor pair and ranks the population

    Parameters
    ----------
    population: list
        list of NFAs with the indices T1, C, T2

    Return
    ------
    ranked_population: nested list
        lists of NFAs and their PCE ranked in order of highest PCE first. Also contains the best donor
        [ranked_NFA_names, ranked_PCE, ranked_best_donor]
    """

    # Best acc/don pair from GAs 1-4 to use in first subcell
    best_acc_abs_filename = '../GA1234/acc_abs_csvs_smear08/39_126_39_abs.csv'
    best_don_abs_filename = '../GA1234/don_abs_csvs_smear08/100_99_111101_abs.csv'

    subcell1_acc_abs =  pd.read_csv(best_acc_abs_filename, index_col=0)
    subcell1_acc_spectraNM = subcell1_acc_abs['nm']
    subcell1_acc_spectraEV = subcell1_acc_abs['eV']
    subcell1_acc_spectraIntensity = subcell1_acc_abs['intensity']

    subcell1_don_abs =  pd.read_csv(best_don_abs_filename, index_col=0)
    subcell1_don_spectraNM = subcell1_don_abs['nm']
    subcell1_don_spectraEV = subcell1_don_abs['eV']
    subcell1_don_spectraIntensity = subcell1_don_abs['intensity']

    pairs = []
    combined_abs_values= []
    overlap_values= []
    scores= []

    for x in range(len(population)):
        filename = utils.make_filename(population[x])

        acceptor_index = population[x][0]
        donor_index = population[x][1]

        exists = os.path.isfile('predictions_GA5_smear08_normalized/%s.txt' % (filename))
        if exists:
            with open('predictions_GA5_smear08_normalized/%s.txt' % (filename), 'r') as file:
                content = file.readline()
                content_list = content.split()
                combined_abs = float(content_list[1])
                overlap = float(content_list[2])
                score = float(content_list[3])
        else:

            acceptor_name = utils.find_name(acceptor_index, 'acc', unit_list)
            donor_name = utils.find_name(donor_index, 'don', unit_list)


            acc_abs_filename = '../GA1234/acc_abs_csvs_smear08/' + acceptor_name + '_abs.csv'
            don_abs_filename = '../GA1234/don_abs_csvs_smear08/' + donor_name + '_abs.csv'

            acc_abs =  pd.read_csv(acc_abs_filename, index_col=0)
            acc_spectraNM = acc_abs['nm']
            acc_spectraEV = acc_abs['eV']
            acc_spectraIntensity = acc_abs['intensity']

            don_abs =  pd.read_csv(don_abs_filename, index_col=0)
            don_spectraNM = don_abs['nm']
            don_spectraEV = don_abs['eV']
            don_spectraIntensity = don_abs['intensity']

            overlap = calc_overlap(subcell1_acc_spectraIntensity, subcell1_don_spectraIntensity,  acc_spectraIntensity, don_spectraIntensity, don_spectraNM)
            combined_abs = calc_combined_abs(subcell1_acc_spectraIntensity, subcell1_don_spectraIntensity,  acc_spectraIntensity, don_spectraIntensity, don_spectraNM)

            #score = combined_abs - overlap

            # normalized based on running this GA once without normalization
            score = ((combined_abs-539.362743945313)/647.264) - ((overlap-238.167)/748.9841)

            # standardized
            #score = ((combined_abs-449.404)/48.1159) - ((overlap-171.3718)/38.336)

            with open('predictions_GA5_smear08_normalized/%s.txt' % (filename), 'w') as file:
                file.write('%s %s %s %s' %(filename, combined_abs, overlap, score))

        pairs.append(population[x])
        combined_abs_values.append(combined_abs)
        overlap_values.append(overlap) 
        scores.append(score)

    ranked_pairs= []
    ranked_combined_abs_values= []
    ranked_overlap_values= []
    ranked_scores= []

    # make list of indicies of NFAs in population, sorted based on PCE
    ranked_indices = list(np.argsort(scores))
    # reverse list so highest property value = 0th
    ranked_indices.reverse()

    # makes list of each property in order of highest PCE pair to lowest
    for x in ranked_indices:
        ranked_pairs.append(pairs[x])
        ranked_combined_abs_values.append(combined_abs_values[x])
        ranked_overlap_values.append(overlap_values[x])
        ranked_scores.append(scores[x])

    ranked_population = [ranked_pairs, ranked_combined_abs_values, ranked_overlap_values, ranked_scores]

    return ranked_population


    



