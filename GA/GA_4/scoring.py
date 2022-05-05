import numpy as np
import pandas as pd

import sklearn
print(sklearn.__version__)

from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit import Chem
from rdkit.Chem import DataStructs
from numpy import linalg
from pickle import load
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import os
import utils

def parse_GFN2(filename):
    '''
    Parses through GFN2-xTB output files

    Parameters
    -----------
    filename: str
        path to output file

    Returns
    -------
    outputs: list
        [dipole_moment, polarizability]
    '''
    
    with open(filename, 'r', encoding = 'utf-8') as file:
        line = file.readline()
        while line:
            if 'molecular dipole' in line:
                line = file.readline()
                line = file.readline()
                line = file.readline()
                    
                line_list = line.split()
                dipole_moment = float(line_list[-1])
                
            elif 'Mol. C8AA' in line:
                line = file.readline()
                line_list = line.split()
                
                polarizability = float(line_list[-1])

            line = file.readline()  
        line = file.readline()

        outputs = [dipole_moment, polarizability]
        
        return outputs
    
def parse_sTDA(filename, HOMO_LUMO=False):
    
    with open(filename, 'r', encoding = 'utf-8') as file:
        line = file.readline()
        oscs = []
        wavelength = []
        energyEV = []
        while line:
            if 'ordered frontier orbitals' in line:
                for x in range(11):
                    line = file.readline()
                line_list = line.split()
                HOMOminus1 = float(line_list[1])
                
                line = file.readline()
                line_list = line.split()
                HOMO = float(line_list[1])
                
                line = file.readline()
                line = file.readline()
                line_list = line.split()
                LUMO = float(line_list[1])
                line = file.readline()
                line_list = line.split()
                LUMOplus1 = float(line_list[1])

                deltaHOMO = abs(HOMOminus1 - HOMO)
                deltaLUMO = abs(LUMO - LUMOplus1)
                fundbg = abs(HOMO-LUMO)

            elif 'excitation energies, transition moments and TDA amplitudes' in line:
                line = file.readline()
                line = file.readline()
                line_list = line.split()
                while line != '\n':
                    line_list = line.split()
                    oscs.append(float(line_list[3]))
                    wavelength.append(float(line_list[2]))
                    energyEV.append(float(line_list[1]))
                    line = file.readline()

            line = file.readline()  
        line = file.readline()

    if HOMO_LUMO == True:
        return HOMO, LUMO
    else:
        
        chemical_potential = (HOMO + LUMO)/2
        hardness =  LUMO - HOMO
        # https://xtb-docs.readthedocs.io/en/latest/sp.html#global-electrophilicity-index
        electrophilicity = chemical_potential**2 / 2*hardness
    
        if len(oscs) != 0:
            summed_oscs = np.sum(oscs)
            
            highest_oscs = 0.0
            opt_bg = round(energyEV[0], 2)
            
            # Opt bg is the energy of the first transition within the first 12 transition with an oscillator strength greater than 0.5 
            if len(oscs) < 12:
                for i in range(len(oscs)):
                    if  oscs[i] > 0.5:
                        opt_bg = round(energyEV[i], 2)
                        break
            else:
                for x in range(12):
                    if  oscs[x] > 0.5:
                        opt_bg = round(energyEV[x], 2)
                        break

            # max abs is the tallest peak in the spectrum
            for x in range(len(oscs)):
                if  oscs[x] > highest_oscs:
                        highest_oscs = oscs[x]
                        max_abs = wavelength[x]
                        
            # Creates full spectrum
            (spectraEV, spectraNM, spectraIntensity) = spectra(energyEV, oscs)
            
            # Calculates the area under the curve using trapz rule for integration
            area_spectra = np.trapz(np.flip(spectraIntensity), np.flip(spectraNM), dx=0.1, axis=- 1)
            
            # Calculates the area under the curve of the simulated spectrum multiplied by the normalized solar spectrum
            area_sim_solar_spectra = solar_integrated_desc(spectraNM, spectraIntensity)
            
            outputs = [HOMO, HOMOminus1, LUMO, LUMOplus1, fundbg, deltaHOMO, deltaLUMO, opt_bg, max_abs, summed_oscs, area_spectra, area_sim_solar_spectra, chemical_potential, electrophilicity]

            return outputs, spectraEV, spectraNM, spectraIntensity
    
        else:
            print(filename)
            print('something is wrong')

def spectra(etens, etoscs, low = 0.5, high = 10.0, resolution = 0.01, smear = 0.04):
    """
    Return arrays of the energies and intensities of a Lorentzian-blurred spectrum

    Parameters
    ----------
    etens: list
        list of transition energies in units of eV
    etoscs: list
        list of oscillator strengths
    low: float
        transition in eV to start spectrum at
    high: float
        transition in eV to end spectrum at
    resolution: float
        increments of eV for spectrum
    smear: float
        blurs intensities of peaks across 0.04 eV

    Returns
    -------
    Lists of the spectra in eV, nm, and their oscillator strengths
    """
    maxSlices = int((high - low) / resolution) + 1
    peaks = len(etens)

    spectraEV = []
    spectraNM = []
    spectraIntensity = []
    for i in range(0, maxSlices):
        energy = float(i * resolution + low) # units of eV
        wavelength = 1239.84193 / energy # convert eV to nm
        intensity = 0.0

        for trans in range(0, peaks):
            this_smear = smear / 0.2 * (-0.046 * etoscs[trans] + 0.20)
            deltaE = etens[trans] - energy
            intensity = intensity + etoscs[trans] * this_smear**2 / (deltaE**2 + this_smear**2)

        spectraEV.append(energy)
        spectraNM.append(wavelength) 
        spectraIntensity.append(intensity)
        
    return spectraEV, spectraNM, spectraIntensity

def custom_round(x, base=5):
    return float(base * round(float(x)/base))

def solar_integrated_desc(spectraNM, spectraIntensity):
    
    wavelength15AM = [] #wavelength for 1.5 AM spectra
    normalized_irr_15AM = []
    
    solar = pd.read_csv('../Solar_radiation_spectrum.csv', index_col = 'wavelength')
    new_spectrum_intensities = []
    
    # the 1.5AM solar spectra does not have constant increments of wavelengths
    for x in range(len(spectraNM)):
        
        if 280 <= spectraNM[x] < 400:
            int_wavelength = custom_round(spectraNM[x], 0.5)
        if 400 <= spectraNM[x] < 1700:
            int_wavelength = custom_round(spectraNM[x], 1)
        if 1700 <= spectraNM[x] < 1702:
            int_wavelength = custom_round(spectraNM[x], 2)
        if 1702 <= spectraNM[x] <=4000:
            int_wavelength = custom_round(spectraNM[x], 5)

        solar_intensity = solar.loc[int_wavelength][-1]
        
        new_spectrum_intensities.append(float(solar_intensity) * spectraIntensity[x])
        
    area_altered_spectra = np.trapz(np.flip(new_spectrum_intensities), np.flip(spectraNM), dx=0.1, axis=- 1)
    
    return area_altered_spectra

def acc_abs_spectra(acceptor):
    with open(acceptor, 'r', encoding = 'utf-8') as file:
        line = file.readline()
        oscs = []
        energyEV = []
        while line:
            if 'excitation energies, transition moments and TDA amplitudes' in line:
                line = file.readline()
                line = file.readline()
                line_list = line.split()
                while line != '\n':
                    line_list = line.split()
                    oscs.append(float(line_list[3]))
                    energyEV.append(float(line_list[1]))
                    line = file.readline()
            line = file.readline()  
        line = file.readline()
    
    # Creates full spectrum
    (acc_spectraEV, acc_spectraNM, acc_spectraIntensity) = spectra(energyEV, oscs)

    return acc_spectraEV, acc_spectraNM, acc_spectraIntensity


def overlap(don_spectraIntensity, acceptor_name):
    
    acceptor = pd.read_csv('acc_abs_csvs/' + acceptor_name + '_abs.csv')
    
    acc_spectraIntensity = acceptor['intensity']
    acc_spectraNM = acceptor['nm']
    
    overlapped_spectra_intensities = [don_spectraIntensity[i] * acc_spectraIntensity[i] for i in range(len(don_spectraIntensity))]
    
    area_altered_spectra = np.trapz(np.flip(overlapped_spectra_intensities), np.flip(acc_spectraNM), dx=0.1, axis=- 1)    
        
    return area_altered_spectra

def getPiSystemSize(mol):
    mol = AllChem.RemoveHs(mol)
    AllChem.Kekulize(mol)
    pi_systems = [pi_system(mol,x.GetIdx(),[x.GetIdx()]) for x in mol.GetAtoms()]
    largest_pi_system = max(pi_systems, key=lambda coll: len(coll))
    pi_system_size = len(largest_pi_system)
    return pi_system_size

def pi_system(mol, current, seen):
    atom = mol.GetAtomWithIdx(current)
    for neighbor in atom.GetNeighbors():
        if (neighbor.GetIdx() not in seen) and (mol.GetBondBetweenAtoms(atom.GetIdx(),neighbor.GetIdx()).GetIsConjugated() or mol.GetBondBetweenAtoms(atom.GetIdx(),neighbor.GetIdx()).GetBondTypeAsDouble() > 1):
            seen.append(neighbor.GetIdx())
            pi_system(mol,neighbor.GetIdx(),seen)
    return seen

def pi_sys_size(filename):
    mol = AllChem.MolFromMolFile(filename)
    pi_size = getPiSystemSize(mol)

    return pi_size

def GetBestFitPlane(pts, weights=None):
    # number of atoms
    wSum = len(pts)
    # sets the origin as the sum of the coordinates for x, y, and z
    origin = np.sum(pts, 0)
    # finds the average of each coordinate and sets as the origin
    origin /= wSum

    # initiates blank coordinates
    sumXX = 0
    sumXY = 0
    sumXZ = 0
    sumYY = 0
    sumYZ = 0
    sumZZ = 0
    sums = np.zeros((3, 3), np.double)
    
    # finds the distance of each point to origin
    for pt in pts:
        # finds the distance of each point to origin
        dp = pt - origin
        
        # sets the 3x3 matrix
        for i in range(3):
            sums[i, i] += dp[i] * dp[i]
            for j in range(i + 1, 3):
                sums[i, j] += dp[i] * dp[j]
                sums[j, i] += dp[i] * dp[j]
    # Averages each number in matrix by the total number of atoms
    sums /= wSum
    
    # Finds the eigenvalues and eigenvectors 
    vals, vects = linalg.eigh(sums)

    # gives indices sorted from smallest to largest
    order = np.argsort(vals)
    
    # smallest eigenvector
    normal = vects[:, order[0]]    
    
    # sets plane coordinates
    plane = np.zeros((4, ), np.double)
    plane[:3] = normal
    plane[3] = -1 * normal.dot(origin)
    
    return plane


def PBFRD(mol, largest_pi_system, confId=-1):
    conf = mol.GetConformer(confId)
    if not conf.Is3D():
        return 0
    
    pts = np.array([list(conf.GetAtomPosition(x)) for x in largest_pi_system])
    plane = GetBestFitPlane(pts)
    
    #distance to point
    denom = np.dot(plane[:3], plane[:3])
    denom = denom**0.5
    # add up the distance from the plane for each point:
    res = 0.0
    for pt in pts:
        res += np.abs(pt.dot(plane[:3]) + plane[3])
        
    res /= denom
    res /= len(pts)
    
    # higher the number, the less planar it is
    return res

def planarity (filename):
    mol = Chem.MolFromMolFile(filename)
    mol = Chem.RemoveHs(mol)
    Chem.Kekulize(mol)
    pi_systems = [pi_system(mol,x.GetIdx(),[x.GetIdx()]) for x in mol.GetAtoms()]
    largest_pi_system = max(pi_systems, key=lambda coll: len(coll))

    planarity = PBFRD(mol, largest_pi_system)

    return planarity

def rdkit_descriptors(filename):
    mol = Chem.MolFromMolFile(filename)
    num_rot_bonds = Descriptors.NumRotatableBonds(mol)
    MolLogP = Descriptors.MolLogP(mol)
    TPSA = Descriptors.TPSA(mol)
    NumHAcceptors = Descriptors.NumHAcceptors(mol)
    NumHDonors = Descriptors.NumHDonors(mol)
    RingCount = Descriptors.RingCount(mol)

    outputs = [num_rot_bonds, MolLogP, TPSA, NumHAcceptors, NumHDonors, RingCount]
    
    return outputs
    
def numpy_2_fp(array):
    fp = DataStructs.cDataStructs.UIntSparseIntVect(len(array))
    for ix, value in enumerate(array):
        fp[ix] = int(value)
    return fp

def morgan_fp_counts(filename):
    mol = Chem.MolFromMolFile(filename)
    fp3 = AllChem.GetHashedMorganFingerprint(mol, 2, nBits=2048)
    array = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp3, array)
    
    fp4 = numpy_2_fp(array)
    
    outputs = list(fp4)

    return outputs

def solvation(filename):
    with open(filename, 'r', encoding = 'utf-8') as file:
        line = file.readline()
        while line:
            if '-> Gsolv' in line:
                line_list = line.split()
                solvation_energy = float(line_list[3])
                break
                
            line = file.readline()  
        line = file.readline()        

    return solvation_energy


def PCE_prediction(population):
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

    # load the model
    model = load(open('../ensemble_rf_ann_mfcounts_standardized_above10_fixed.pkl', 'rb'))
    # load the scaler
    scaler = load(open('../scaler_fixed.pkl', 'rb'))

    acceptors = pd.read_csv('acc_descriptors_GA123_top200.csv')

    # create descriptors
    column_names = ['A-HOMO', 'A-HOMOminus1', 'A-LUMO', 'A-LUMOplus1', 'A-fundbg', 'A-deltaHOMO', 'A-deltaLUMO', 'A-opt_bg', 'A-max_abs', 'A-summed_oscs', 'A-area_spectra', 'A-area_sim_solar_spectra', 'A-chemical_potential', 'A-electrophilicity', 'A-pi_sys_size', 'A-num_rot_bonds', 'A-MolLogP', 'A-TPSA', 'A-NumHAcceptors', 'A-NumHDonors', 'A-RingCount','A-planarity','A-dipole_moment', 'A-polarizability', 'A-SolvationEnergy_water', 'A-SolvationEnergy_hexane', 'D-HOMO', 'D-HOMOminus1', 'D-LUMO', 'D-LUMOplus1', 'D-fundbg', 'D-deltaHOMO', 'D-deltaLUMO', 'D-opt_bg', 'D-max_abs', 'D-summed_oscs', 'D-area_spectra', 'D-area_sim_solar_spectra', 'D-chemical_potential', 'D-electrophilicity', 'D-pi_sys_size', 'D-num_rot_bonds', 'D-MolLogP', 'D-TPSA', 'D-NumHAcceptors', 'D-NumHDonors', 'D-RingCount',  'D-planarity','D-dipole_moment', 'D-polarizability', 'D-SolvationEnergy_water', 'D-SolvationEnergy_hexane', 'AD-overlap', 'AD-HOMOoffset', 'AD-LUMOoffset', 'DHOMO_ALUMO_offset']
    
    fp_column_names = []
    # add column names for 2048 bit morgan fingerprints
    for x in range(2048):
        col_name = 'A-ECFP_' + str(x)
        fp_column_names.append(col_name)
    for x in range(2048):
        col_name = 'D-ECFP_' + str(x)
        fp_column_names.append(col_name)

    PCE_list = []
    NFA_name = []
    best_acceptors = []
    NFA_population = []

    for x in range(len(population)):
        filename = utils.make_file_name(population[x])

        exists = os.path.isfile('PCE_predictions_GA4/%s.txt' % (filename))
        if exists:
            with open('PCE_predictions_GA4/%s.txt' % (filename), 'r') as file:
                content = file.readline()
                content_list = content.split()
                best_PCE = float(content_list[1])
                best_acceptor = content_list[2]

        else:
            try:

                don_stda = 'sTDDFTxtb_output/' + filename + '.stda'
                don_mol = 'GFN2_output/' + filename + '.mol'
                don_GFN2 = 'GFN2_output/' + filename + '.out'
                don_solv_water = 'solvation_water/' + filename + '.out'
                don_solv_hexane = 'solvation_hexane/' + filename + '.out'

                # parse sTDDFT-xtb output files of acceptors
                sTDDFTxtb_output, don_spectraEV, don_spectraNM, don_spectraIntensity = parse_sTDA(don_stda) #HOMO, HOMOminus1, LUMO, LUMOplus1, fundbg, deltaHOMO, deltaLUMO, opt_bg, max_abs, summed_oscs, area_spectra, area_sim_solar_spectra, chemical_potential, electrophilicity
                
                don_HOMO = sTDDFTxtb_output[0]
                don_LUMO = sTDDFTxtb_output[2]

                # parse GFN2-xtb files for acceptor
                pi_size = pi_sys_size(don_mol) #pi_size
                rdkit_desc = rdkit_descriptors(don_mol) #num_rot_bonds, MolLogP, TPSA, NumHAcceptors, NumHDonors
                plan = planarity(don_mol) # planarity

                # calculate pi system size of acceptor
                GFN2 = parse_GFN2(don_GFN2) #dipole_moment, polarizability

                # calculate solvation free energy of acceptor in water
                solv_water = solvation(don_solv_water) #solvation_energy

                # calculate solvation free energy of acceptor in hexane
                solv_hexane = solvation(don_solv_hexane) #solvation_energy

                # For morgan fingerprint counts
                don_fp = morgan_fp_counts(don_mol)

                best_PCE = 0
                
                for i in range(len(acceptors)):

                    # precalculated donor properties
                    acc_props = list(acceptors.iloc[i])[2:28]

                    acc_HOMO = acc_props[0]
                    acc_LUMO = acc_props[2]

                    # makes sure the energy levels are aligned so the donor acts as a donor
                    if don_HOMO > acc_HOMO and don_LUMO > acc_LUMO:

                        overlap_AD = overlap(don_spectraIntensity, list(acceptors.iloc[i])[1])

                        AD_HOMOoffset = sTDDFTxtb_output[0] - acc_props[0]
                        AD_LUMOoffset = sTDDFTxtb_output[2] - acc_props[2]
                        DHOMO_ALUMO_offset = acc_props[2] - sTDDFTxtb_output[0]

                        data = []
                        data.extend(acc_props)

                        data.extend(sTDDFTxtb_output)
                        data.append(pi_size)
                        data.extend(rdkit_desc)
                        data.append(plan)
                        data.extend(GFN2)
                        data.append(solv_water)
                        data.append(solv_hexane)
                        data.append(overlap_AD)
                        data.append(AD_HOMOoffset)
                        data.append(AD_LUMOoffset)
                        data.append(DHOMO_ALUMO_offset)

                        df = pd.DataFrame(data).T
                        df.columns = column_names
                        
                        # scale the data. Uses scaler from OPEP2
                        df[column_names] = scaler.transform(df[column_names])
                        
                        fp = []

                        # add morgan fingerprint counts for acceptor
                        fp.extend((list(acceptors.iloc[i])[28:]))
                        # add morgan fingerprint counts for donor
                        fp.extend(don_fp)

                        
                        df_fp = pd.DataFrame(fp).T
                        df_fp.columns = fp_column_names
                        
                        data_with_fp = pd.concat([df, df_fp], axis=1)
                    
                        PCE = float(model.predict(data_with_fp))

                    else:
                        PCE = 0


                    if PCE > best_PCE:
                        best_PCE = PCE
                        best_acceptor = list(acceptors.iloc[i])[1]
            except:
                best_PCE = 0
                best_acceptor = "bad file"

            with open('PCE_predictions_GA4/%s.txt' % (filename), 'w') as file:
                file.write('%s %s %s' %(filename, best_PCE, best_acceptor))

        PCE_list.append(best_PCE)
        NFA_name.append(filename)
        best_acceptors.append(best_acceptor)
        NFA_population.append(population[x]) 

    ranked_PCE = []
    ranked_NFA_names = []
    ranked_best_acceptor = []
    ranked_NFA_population = []

    # make list of indicies of NFAs in population, sorted based on PCE
    ranked_indices = list(np.argsort(PCE_list))
    # reverse list so highest property value = 0th
    ranked_indices.reverse()

    # makes list of each property in order of highest PCE pair to lowest
    for x in ranked_indices:
        ranked_PCE.append(PCE_list[x])
        ranked_NFA_names.append(NFA_name[x])
        ranked_best_acceptor.append(best_acceptors[x])
        ranked_NFA_population.append(NFA_population[x])

    ranked_population = [ranked_NFA_names, ranked_PCE, ranked_best_acceptor, ranked_NFA_population]

    return ranked_population


    



