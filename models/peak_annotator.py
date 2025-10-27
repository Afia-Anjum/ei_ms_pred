import numpy as np
import scipy.signal as cg
from rdkit import Chem
import sys

import ChEMBL_Structure_Pipeline
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from PyCFMID.PyCFMID import fraggraph_gen
import re
#from PyCFMID.PyCFMID import *
import itertools

import os
import platform
import json
import requests
import pubchempy as pc
from bs4 import BeautifulSoup
import subprocess
import pandas as pd
import numpy as np
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


def check_output_file(output_file=None):
    if output_file is None:
        try:
            os.mkdir('Output')
        except:
            pass
        output_file = os.path.join(os.getcwd(), 'Output', 'output.txt')
    return output_file


def check_input_file(input_dir=None):
    if input_dir is None:
        try:
            os.mkdir('Input')
        except:
            pass
        input_dir = os.path.join(os.getcwd(), 'Input')
    return input_dir


def fraggraph_gen(smiles, max_depth=2, ionization_mode='+', fullgraph=True, output_file=None):
    output_file = check_output_file(output_file)
    os.chdir('D:\\NIST20\\MSSEARCH\\pythonProject\\')
    program = os.path.join('PyCFMID', platform.platform().split('-')[0], 'fraggraph-gen.exe')
    cmd = os.path.join(os.getcwd(), program)
    cmd += ' ' + str(smiles)
    cmd += ' ' + str(max_depth)
    cmd += ' ' + str(ionization_mode)
    if fullgraph:
        cmd += ' fullgraph'
    else:
        cmd += ' fragonly'
    cmd += ' ' + str(output_file)
    subprocess.call(cmd)
    return parser_fraggraph_gen(output_file)


def parser_fraggraph_gen(output_file):
    with open(output_file) as t:
        output = t.readlines()
    output = [s.replace('\n', '') for s in output]
    if len(output)==0:
        return None
    nfrags = int(output[0])
    frag_index = [output[i].split(' ')[0] for i in range(1, nfrags + 1)]
    frag_mass = [output[i].split(' ')[1] for i in range(1, nfrags + 1)]
    frag_smiles = [output[i].split(' ')[2] for i in range(1, nfrags + 1)]
    loss_from = [output[i].split(' ')[0] for i in range(nfrags + 2, len(output))]
    loss_to = [output[i].split(' ')[0] for i in range(nfrags + 2, len(output))]
    loss_smiles = [output[i].split(' ')[0] for i in range(nfrags + 2, len(output))]
    fragments = pd.DataFrame({'index': frag_index, 'mass': frag_mass, 'smiles': frag_smiles})
    losses = pd.DataFrame({'from': loss_from, 'to': loss_to, 'smiles': loss_smiles})
    return {'fragments': fragments, 'losses': losses}


def cfm_predict(smiles, prob_thresh=0.001, param_file='', config_file='', annotate_fragments=False, output_file=None,
                apply_postproc=True, suppress_exceptions=False):
    output_file = check_output_file(output_file)
    if param_file == '':
        param_file = os.path.join(os.getcwd(), 'PyCFMID', 'Pretrain', 'param_output0.log')
    if config_file == '':
        config_file = os.path.join(os.getcwd(), 'PyCFMID', 'Pretrain', 'param_config.txt')
    program = os.path.join('PyCFMID', platform.platform().split('-')[0], 'cfm-predict.exe')
    cmd = os.path.join(os.getcwd(), program)
    cmd += ' ' + smiles
    cmd += ' ' + str(prob_thresh)
    cmd += ' ' + param_file
    cmd += ' ' + config_file
    if annotate_fragments:
        cmd += ' ' + str(1)
    else:
        cmd += ' ' + str(0)
    cmd += ' ' + output_file
    if apply_postproc:
        cmd += ' ' + str(1)
    else:
        cmd += ' ' + str(0)
    if suppress_exceptions:
        cmd += ' ' + str(1)
    else:
        cmd += ' ' + str(0)
    subprocess.call(cmd)
    return parser_cfm_predict(output_file)

def parser_cfm_predict(output_file):
    with open(output_file) as t:
        output = t.readlines()
    output = [s.replace('\n', '') for s in output]
    low_energy = pd.DataFrame(columns=['mz', 'intensity'])
    medium_energy = pd.DataFrame(columns=['mz', 'intensity'])
    high_energy = pd.DataFrame(columns=['mz', 'intensity'])
    energy_level = 0
    for i in output:
        if 'energy0' == i:
            energy_level = 0
        elif 'energy1' == i:
            energy_level = 1
        elif 'energy2' == i:
            energy_level = 2
        elif '' == i:
            continue
        else:
            i = i.split(' ')
            i = [float(j) for j in i]
            if energy_level == 0:
                low_energy.loc[len(low_energy)] = i
            elif energy_level == 1:
                medium_energy.loc[len(medium_energy)] = i
            else:
                high_energy.loc[len(high_energy)] = i
    return {'low_energy': low_energy, 'medium_energy': medium_energy, 'high_energy': high_energy}


def cfm_id(spectrum_file, candidate_file, num_highest=-1, ppm_mass_tol=10, abs_mass_tol=0.01, prob_thresh=0.001,
           param_file='', config_file='', score_type='Jaccard', apply_postprocessing=True, output_file=None):
    output_file = check_output_file(output_file)
    if param_file == '':
        param_file = os.path.join(os.getcwd(), 'PyCFMID', 'Pretrain', 'param_output0.log')
    if config_file == '':
        config_file = os.path.join(os.getcwd(), 'PyCFMID', 'Pretrain', 'param_config.txt')
    program = os.path.join('PyCFMID', platform.platform().split('-')[0], 'cfm-id.exe')
    cmd = os.path.join(os.getcwd(), program)
    cmd += ' ' + spectrum_file
    cmd += ' ' + 'AN_ID'
    cmd += ' ' + candidate_file
    cmd += ' ' + str(num_highest)
    cmd += ' ' + str(ppm_mass_tol)
    cmd += ' ' + str(abs_mass_tol)
    cmd += ' ' + str(prob_thresh)
    cmd += ' ' + param_file
    cmd += ' ' + config_file
    cmd += ' ' + score_type
    if apply_postprocessing:
        cmd += ' ' + str(1)
    else:
        cmd += ' ' + str(0)
    cmd += ' ' + output_file
    subprocess.call(cmd)
    return parser_cfm_id(output_file)


def cfm_id_database(spectrum_dataframe, formula, energy_level='high', database='biodb', input_dir=None, num_highest=-1,
                    ppm_mass_tol=10, abs_mass_tol=0.01, prob_thresh=0.001, param_file='', config_file='',
                    score_type='Jaccard', apply_postprocessing=True, output_file=None):
    input_dir = check_input_file(input_dir)
    output_file = check_output_file(output_file)
    if param_file == '':
        param_file = os.path.join(os.getcwd(), 'PyCFMID', 'Pretrain', 'param_output0.log')
    if config_file == '':
        config_file = os.path.join(os.getcwd(), 'PyCFMID', 'Pretrain', 'param_config.txt')
    spectrum_file = os.path.join(input_dir, 'spectrum.txt')
    candidate_file = os.path.join(input_dir, 'candidate.txt')
    spectrum_file = write_spectrum(spectrum_dataframe, spectrum_file, energy_level)
    if database == 'biodb':
        candidates = search_biodatabase(formula, candidate_file)
    elif database == 'database':
        candidates = search_pubchem(formula, candidate_file)
    else:
        candidates = pd.read_csv(database)
        output = pd.DataFrame({'ID': candidates.index, 'Smiles': candidates['SMILES']})
        output.to_csv(candidate_file, header=False, index=False, sep=' ')
    result = cfm_id(spectrum_file, candidate_file, num_highest, ppm_mass_tol, abs_mass_tol, prob_thresh, param_file,
                    config_file, score_type, apply_postprocessing, output_file)
    return {'candidates': candidates, 'result': result}

def write_spectrum(spectrum_dataframe, spectrum_file, energy_level='high'):
    with open(spectrum_file, 'w+') as t:
        t.write('energy0' + '\n')
        if energy_level == 'low':
            for s in range(len(spectrum_dataframe)):
                t.write(str(spectrum_dataframe.iloc[s, 0]) + ' ' + str(spectrum_dataframe.iloc[s, 1]) + '\n')
        t.write('energy1' + '\n')
        if energy_level == 'medium':
            for s in range(len(spectrum_dataframe)):
                t.write(str(spectrum_dataframe.iloc[s, 0]) + ' ' + str(spectrum_dataframe.iloc[s, 1]) + '\n')
        t.write('energy2' + '\n')
        if energy_level == 'high':
            for s in range(len(spectrum_dataframe)):
                t.write(str(spectrum_dataframe.iloc[s, 0]) + ' ' + str(spectrum_dataframe.iloc[s, 1]) + '\n')
    return spectrum_file


def parser_cfm_id(output_file):
    output = pd.read_table(output_file, delim_whitespace=True, header=None, index_col=0)
    output.columns = ['Score', 'ID', 'Smiles']
    return output


structureDB = pd.read_table('PyCFMID/Database/MsfinderStructureDB-VS12.esd')


def search_biodatabase(formula, output_file=None):
    output_file = check_output_file(output_file)
    result = structureDB[structureDB['Formula'] == formula]
    output = pd.DataFrame({'ID': result.index, 'Smiles': result['SMILES']})
    output.to_csv(output_file, header=False, index=False, sep=' ')
    return result


def search_pubchem(formula, output_file=None, timeout=999):
    output_file = check_output_file(output_file)
    # get pubchem cid based on formula
    cids = pc.get_cids(formula, 'formula', list_return='flat')
    idstring = ''
    smiles = []
    inchikey = []
    all_cids = []
    # search pubchem via formula with pug
    for i, cid in enumerate(cids):
        idstring += ',' + str(cid)
        if ((i % 100 == 99) or (i == len(cids) - 1)):
            url_i = "http://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/" + idstring[1:(
                len(idstring))] + "/property/InChIKey,CanonicalSMILES/JSON"
            res_i = requests.get(url_i, timeout=timeout)
            soup_i = BeautifulSoup(res_i.content, "html.parser")
            str_i = str(soup_i)
            properties_i = json.loads(str_i)['PropertyTable']['Properties']
            idstring = ''
            for properties_ij in properties_i:
                smiles_ij = properties_ij['CanonicalSMILES']
                if smiles_ij not in smiles:
                    smiles.append(smiles_ij)
                    inchikey.append(properties_ij['InChIKey'])
                    all_cids.append(str(properties_ij['CID']))
                else:
                    wh = np.where(np.array(smiles) == smiles_ij)[0][0]
                    all_cids[wh] = all_cids[wh] + ', ' + str(properties_ij['CID'])
    result = pd.DataFrame({'InChIKey': inchikey, 'SMILES': smiles, 'PubChem': all_cids})
    output = pd.DataFrame({'ID': result.index, 'Smiles': result['SMILES']})
    output.to_csv(output_file, header=False, index=False, sep=' ')
    return result


Smarts_Ketone="[#6][CX3](=O)[#6]"
Smarts_Aldehyde="[CX3H1](=O)[#6]"
Smarts_CarboxylicAcid="[CX3](=O)[OX2H1]"
Smarts_Amides="[NX3][CX3](=[OX1])[#6]"
Smarts_Nitrile="[NX1]#[CX2]"
Smarts_Aromatic="a"

# mass of elements(in case of isotopes, the most abundant one is taken)
mass_dict = {'C': 12, 'H': 1, 'N': 14, 'P': 31, 'Si': 28, 'S': 32, 'O': 16, 'Cl': 35, 'F': 19, 'I': 127, 'Br': 79, 'Na': 23, 'Gd':157,'Nb':93,'Xe':131,'Ar':40,'Rh':103,'V':51,'Cu':63,'Mn':55,'Mo':96,'Zr':91,'Sr':87,'Pt':195,'Zn':65,'Ca':40,'Y':89,'Ga':69,'Pd':106,'Mg':24,'Ti':48,'Cr':52,'Sb':121,'Pr':141,'Co':59,'Hg':200,'Pb':207,'K':39,'Cd':112,'Ge':72,'Li':6,'Eu':151,'Te':127,'In':114,'Ag':107,'W':183,'Re':186,'As':74,'Be':9,'Ce':140,'Sn':118,'Ru':101,'Ni':58,'La':138,'Al':26,'Yb':173,'Rb':85,'Os':190,'Sc':44,'Kr':83,'Tl':204,'Fe':55,'Se':79,'Bi':209,'B':10,'Ir':192,'Th':232,'Cs':133,'U':238 }
# will take the ground state valence only
valency_dict = {'C': 4, 'H': 1, 'N': 3, 'P': 3, 'Si': 4, 'S': 2, 'O': 2, 'Cl': 1, 'F': 1, 'I': 1, 'Br': 1, 'Na': 1,'Gd':3,'Nb':5,'Xe':0,'Ar':0,'Rh':6,'V':5,'Cu':3,'Mn':7,'Mo':6,'Zr':4,'Sr':2,'Pt':6,'Zn':2,'Ca':2,'Y':3,'Ga':3,'Pd':6,'Mg':2,'Ti':4,'Cr':6,'Sb':5,'Pr':3,'Co':4,'Hg':2,'Pb':4,'K':1,'Cd':2,'Ge':4,'Li':1,'Eu':3,'Te':6,'In':3,'Ag':3,'W':6,'Re':7,'As':5,'Be':2,'Ce':4,'Sn':4,'Ru':8,'Ni':4,'La':3,'Al':3,'Yb':3,'Rb':1,'Os':8,'Sc':3,'Kr':0,'Tl':3,'Fe':6,'Se':6,'Bi':5,'B':3,'Ir':6,'Th':4,'Cs':4,'U':6}

def coincombination(amount, index, list1, odd, nitro, M, cfmid_frag_mass, cfmid_frag_ions, elements,coins, counting_coins, valency_coins, DOU_M):
    if amount == 0:
        # elemental parent composition
        flag = 0
        for p in range(len(coins)):
            if list1.count(coins[p]) > counting_coins[p]:
                flag = 1
                break
        nitro_rule_flag = 0
        # senior rule
        senior_rule_check1 = 0
        senior_rule_check2 = 0
        senior_rule_check3 = 0

        num_of_atoms = 0  # no of atoms having odd valences count
        No_Atoms = 0
        sum_valency = 0
        max_valency = 0
        for p in range(len(coins)):
            No_Atoms = No_Atoms + 1
            if list1.count(coins[p]) > 0:
                if valency_coins[p] > max_valency:
                    max_valency = valency_coins[p]
                sum_valency = sum_valency + valency_coins[p] * (list1.count(coins[p]))
                if valency_coins[p] % 2 == 1:
                    num_of_atoms = num_of_atoms + list1.count(coins[p])

        if senior_rule_check1 == 1:
            num_of_atoms = 0  # no of atoms having odd valences count
            No_Atoms = 1
            sum_valency = 0
            max_valency = 0
            for p in range(len(coins)):
                No_Atoms = No_Atoms + 1
                if list1.count(coins[p]) > 0:
                    if valency_coins[p] > max_valency:
                        max_valency = valency_coins[p]
                    if coins[p] == 1:
                        sum_valency = sum_valency + valency_coins[p] * (list1.count(coins[p]) + 1)
                    else:
                        sum_valency = sum_valency + valency_coins[p] * (list1.count(coins[p]))
                    if valency_coins[p] % 2 == 1:
                        if coins[p] == 1:
                            num_of_atoms = num_of_atoms + list1.count(coins[p]) + 1
                        else:
                            num_of_atoms = num_of_atoms + list1.count(coins[p])

        # elemental ratios check
        elemental_check_flag = 0
        M_fragments = 0
        for i in range(len(list1)):
            M_fragments = M_fragments + list1[i]

        # degree of unsaturation calculation to get rid of neutral molecules, only keep the charged entities
        DOU_flag = 0
        num_Halide = 0
        num_H = 0
        num_N = 0
        num_C = 0

        for p in range(len(coins)):
            if coins[p] == 12:
                num_C = list1.count(coins[p])
            elif coins[p] == 14:
                num_N = list1.count(coins[p])
            elif coins[p] == 1:
                num_H = list1.count(coins[p])
            elif coins[p] == 35 or coins[p] == 19 or coins[p] == 79 or coins[p] == 127:
                num_Halide = num_Halide + list1.count(coins[p])
                
        DOU = (2 * num_C + 2 + num_N - num_Halide - num_H) / 2  # this is the DOU for each fragment ions(for each coins)
        diff_flag=0
        frac=DOU-int(DOU)
        if DOU >= 0:
            if DOU <= (DOU_M+1):
               diff_flag=1
        if M_fragments!=M:
            if DOU<=0 or diff_flag==0: #or frac==0:
                DOU_flag=1

        if flag == 0 and nitro_rule_flag == 0 and senior_rule_check1 == 0 and senior_rule_check2 == 0 and senior_rule_check3 == 0 and elemental_check_flag == 0:
            if DOU_flag == 0:
                f = open("demofile.txt", "a")
                string1 = ""
                sum_list = sum(list1)
                formula_cfm = ""
                for ms in range(len(cfmid_frag_mass)):
                    if sum_list == cfmid_frag_mass[ms]:
                        mol_cfm = Chem.MolFromSmiles(cfmid_frag_ions[ms])
                        formula_cfm = rdMolDescriptors.CalcMolFormula(mol_cfm)
                        if formula_cfm.endswith("+"):
                            formula_cfm = formula_cfm[:-1]
                for l in range(len(elements)):
                    mass_elements = mass_dict[elements[l]]
                    coefficient_element = list1.count(mass_elements)
                    if coefficient_element > 0:
                        string1 = string1 + elements[l]
                        if coefficient_element > 1:
                            string1 = string1 + str(coefficient_element)

                if formula_cfm:
                    if string1 == formula_cfm:
                        f.write(formula_cfm)
                        f.write(" ")
                        f.close()
                else:
                    f.write(string1)
                    f.write(" ")
                    f.close()
        return
    if amount < 0:
        return
    for i in range(index, len(coins)):
        coin = coins[i]
        if amount >= coin:
            list1.append(coin)
            coincombination(amount - coin, i, list1, odd, nitro, M,cfmid_frag_mass, cfmid_frag_ions, elements,coins, counting_coins,valency_coins, DOU_M)
            list1.pop()


def check_functional_groups(smiles):
    functional_group = [0, 0, 0, 0, 0, 0]
    query_ketone = Chem.MolFromSmarts(Smarts_Ketone)
    query_aldehyde = Chem.MolFromSmarts(Smarts_Aldehyde)
    query_amides = Chem.MolFromSmarts(Smarts_Amides)
    query_nitrile = Chem.MolFromSmarts(Smarts_Nitrile)
    query_carboxylic_acid = Chem.MolFromSmarts(Smarts_CarboxylicAcid)
    query_aromatic = Chem.MolFromSmarts(Smarts_Aromatic)

    #mol = Chem.MolFromSmiles(smiles[0])
    mol = Chem.MolFromSmiles(smiles)
    match_list1 = mol.HasSubstructMatch(query_ketone)
    match_list2 = mol.HasSubstructMatch(query_aldehyde)
    match_list3 = mol.HasSubstructMatch(query_amides)
    match_list4 = mol.HasSubstructMatch(query_nitrile)
    match_list5 = mol.HasSubstructMatch(query_carboxylic_acid)
    match_list6 = mol.HasSubstructMatch(query_aromatic)
    if match_list1:
        functional_group[0] = 1
    if match_list2:
        functional_group[1] = 1
    if match_list3:
        functional_group[2] = 1
    if match_list4:
        functional_group[3] = 1
    if match_list5:
        functional_group[4] = 1
    if match_list6:
        functional_group[5] = 1

    return functional_group

def dict_to_formula(elements_dict):
    formula = ""
    for symbol, count in sorted(elements_dict.items()):
        formula += symbol
        if count > 1:
            formula += str(count)
    return formula

def count_elements(formula):
    # Split the formula into individual element symbols and their counts
    element_counts = re.findall(r'([A-Z][a-z]*)(\d*)', formula)

    # Create a dictionary to store the element counts
    element_dict = {}
    for element, count in element_counts:
        count = int(count) if count else 1
        element_dict[element] = element_dict.get(element, 0) + count

    return element_dict


def guess_formula(smiles_string, mass_array, intensity_array):
    functional_group = check_functional_groups(smiles_string)

    frags = fraggraph_gen(smiles_string, max_depth=2, ionization_mode='*', fullgraph=True)
    if not frags:
        cfmid_frag_ions=[]
        cfmid_frag_mass=[]
    else:
        fg = frags['fragments']
        cfmid_frag_ions = fg['smiles'].values.tolist()
        cfmid_frag_mass = fg['mass'].values.tolist()
        # cfmid_frag_mass_float = [float(i) for i in cfmid_frag_mass]
        cfmid_frag_mass = [int(float(i)) for i in cfmid_frag_mass]

    list2 = []
    # Convert SMILES to a RDKit molecule object
    mol = Chem.MolFromSmiles(smiles_string)

    # Calculate the molecular formula of the molecule
    formula = rdMolDescriptors.CalcMolFormula(mol)
    formula_len = len(formula)

    elemental_composition_dict = count_elements(formula)
    # print(element_counts)  # Output: {'C': 6, 'H': 12, 'O': 6}

    elements = list(elemental_composition_dict.keys())

    # calculate DOU of the molecular ion
    DOU_M = 0
    if 'C' in elemental_composition_dict.keys():
        DOU_M += 2 * elemental_composition_dict['C']
    if 'N' in elemental_composition_dict.keys():
        DOU_M += elemental_composition_dict['N']
    if 'H' in elemental_composition_dict.keys():
        DOU_M -= elemental_composition_dict['H']
    if 'F' in elemental_composition_dict.keys():
        DOU_M -= elemental_composition_dict['F']
    if 'Cl' in elemental_composition_dict.keys():
        DOU_M -= elemental_composition_dict['Cl']
    if 'Br' in elemental_composition_dict.keys():
        DOU_M -= elemental_composition_dict['Br']
    if 'I' in elemental_composition_dict.keys():
        DOU_M -= elemental_composition_dict['I']

    DOU_M = (DOU_M + 2) / 2
    # three variables for the coin change program
    coins = []
    counting_coins = []
    valency_coins = []

    # corresponding atomic mass for the element and sort them
    coin_dict = {}
    for i in range(len(elements)):
        coin_dict[elements[i]] = mass_dict[elements[i]]

    # print(coin_dict)
    coin_dict = {k: v for k, v in sorted(coin_dict.items(), key=lambda item: item[1])}
    for i in range(len(coin_dict)):
        coins.append(coin_dict[list(coin_dict.keys())[i]])
        counting_coins.append(elemental_composition_dict[list(coin_dict.keys())[i]])
        valency_coins.append(valency_dict[list(coin_dict.keys())[i]])
    intensities=intensity_array
    M = 0
    for i in range(len(counting_coins)):
        M = M + coins[i] * counting_coins[i]
    max_M = max(mass_array)
    MW_mz_list = list(range(int(max(mass_array))+1))  # initialize the list with 0's
    MW_intensity_list = [0] * len(MW_mz_list)

    ######commenting cluster peaks lines####
    for i in range(len(mass_array)):
        print(int(mass_array[i]))
        MW_intensity_list[int(mass_array[i])] = int(intensities[i])

    # lists for intensities
    cluster_list = []
    temp_cluster = []

    # lists for m_by_z
    m_by_z_cluster_list = []
    m_by_z_temp_cluster = []

    flag_clust = 0
    for i in range(len(MW_intensity_list)):
        flag_clust = 0
        if MW_intensity_list[i] > 0:
            temp_cluster.append(MW_intensity_list[i])
            m_by_z_temp_cluster.append(MW_mz_list[i])
            flag_clust = 1
        if len(temp_cluster) > 0 and flag_clust == 0:
            cluster_list.append(temp_cluster)
            m_by_z_cluster_list.append(m_by_z_temp_cluster)
            temp_cluster = []
            m_by_z_temp_cluster = []

    if len(temp_cluster) > 0:
        cluster_list.append(temp_cluster)
        m_by_z_cluster_list.append(m_by_z_temp_cluster)

    ######commenting cluster peaks lines####
    new_cluster_list = []
    new_cluster_list_m_by_z = []
    # taking peak values of each cluster and deleting everything below 1 percent of that peak
    for i in range(len(cluster_list)):
        cluster_peak = max(cluster_list[i])
        threshold = cluster_peak * 0.01
        clust_list = cluster_list[i]
        m_by_z_clust_list = m_by_z_cluster_list[i]
        # print(clust_list)
        tmp_clust = []
        tmp_mz_clust = []
        for j in range(len(clust_list)):
            if clust_list[j] > threshold:
                tmp_clust.append(clust_list[j])
                tmp_mz_clust.append(m_by_z_clust_list[j])
        new_cluster_list.append(tmp_clust)
        new_cluster_list_m_by_z.append(tmp_mz_clust)

    # combine back
    new_intensity_list = list(itertools.chain.from_iterable(new_cluster_list))
    new_m_by_z_list = list(itertools.chain.from_iterable(new_cluster_list_m_by_z))


    # reassigning
    local_max_m_by_z = new_m_by_z_list

    f = open("demofile.txt", "w")
    f.close()

    for i in range(len(local_max_m_by_z)):

        manual_writing_flag = 0
        f = open("demofile.txt", "a")
        if i > 0:
            f.write("\n")
        f.write("m/z: " + str(local_max_m_by_z[i]))
        f.write(" ")
        if functional_group[0] == 1 and local_max_m_by_z[i] == 43:
            print("ketone")
        if functional_group[1] == 1:
            if local_max_m_by_z[i] == 44:
                # print("aldehyde")
                f.write("C2H4O ")
                manual_writing_flag = 1
            if local_max_m_by_z[i] == 29:
                f.write("COH ")
                manual_writing_flag = 1
            if local_max_m_by_z[i] == (M - 18):        #loss of water H20
                new_comp_dict=elemental_composition_dict.copy()
                if 'H' in new_comp_dict and 'O' in new_comp_dict:
                    new_comp_dict['H']=new_comp_dict['H']-2
                    new_comp_dict['O'] = new_comp_dict['O'] - 1
                string1=dict_to_formula(new_comp_dict)
                f.write(string1)
                f.write(" ")
                manual_writing_flag = 1
            if local_max_m_by_z[i] == (M - 28):            #loss of C2H4
                new_comp_dict=elemental_composition_dict.copy()
                if 'C' in new_comp_dict and 'H' in new_comp_dict:
                    new_comp_dict['C']=new_comp_dict['C']-2
                    new_comp_dict['H'] = new_comp_dict['H'] - 4
                string1=dict_to_formula(new_comp_dict)
                f.write(string1)
                f.write(" ")
                manual_writing_flag = 1
            if local_max_m_by_z[i] == (M - 44):                 #loss of C2H4O
                new_comp_dict=elemental_composition_dict.copy()
                if 'C' in new_comp_dict and 'H' in new_comp_dict and 'O' in new_comp_dict:
                    new_comp_dict['C']=new_comp_dict['C']-2
                    new_comp_dict['H'] = new_comp_dict['H'] - 4
                    new_comp_dict['O'] = new_comp_dict['O'] - 1
                string1=dict_to_formula(new_comp_dict)
                f.write(string1)
                f.write(" ")
                manual_writing_flag = 1
        if functional_group[2] == 1:
            if local_max_m_by_z[i] == 30:
                f.write("CHN4")
                f.write(" ")
                manual_writing_flag = 1
            if local_max_m_by_z[i] == 44:
                f.write("CH2NO")
                f.write(" ")
                manual_writing_flag = 1
            if local_max_m_by_z[i] == 59:
                f.write("C2H5NO")
                f.write(" ")
                manual_writing_flag = 1
            if local_max_m_by_z[i] == 86:
                f.write("C4H8NO")
                f.write(" ")
                manual_writing_flag = 1
        if functional_group[3] == 1:
            # print("nitrile")
            if local_max_m_by_z[i] == 41:
                f.write("C2H3N")
                f.write(" ")
                manual_writing_flag = 1
            if local_max_m_by_z[i] == (M - 1):
                new_comp_dict=elemental_composition_dict.copy()
                if 'H' in new_comp_dict:
                    new_comp_dict['H'] = new_comp_dict['H'] - 1
                string1=dict_to_formula(new_comp_dict)
                f.write(string1)
                f.write(" ")
                manual_writing_flag = 1
        if functional_group[4] == 1:
            # print("carboxylic acid")
            if local_max_m_by_z[i] == 60:
                f.write("C2H4O2")
                f.write(" ")
                manual_writing_flag = 1
            if local_max_m_by_z[i] == (M - 17):
                new_comp_dict=elemental_composition_dict.copy()
                if 'H' in new_comp_dict and 'O' in new_comp_dict:
                    new_comp_dict['H'] = new_comp_dict['H'] - 1
                    new_comp_dict['O'] = new_comp_dict['O'] - 1
                string1=dict_to_formula(new_comp_dict)
                f.write(string1)
                f.write(" ")
                manual_writing_flag = 1
            if local_max_m_by_z[i] == (M - 45):     #loss of COOH
                new_comp_dict=elemental_composition_dict.copy()
                if 'C' in new_comp_dict and 'O' in new_comp_dict and 'H' in new_comp_dict:
                    new_comp_dict['H'] = new_comp_dict['H'] - 1
                    new_comp_dict['O'] = new_comp_dict['O'] - 2
                    new_comp_dict['C'] = new_comp_dict['C'] - 1
                string1=dict_to_formula(new_comp_dict)
                f.write(string1)
                f.write(" ")
                manual_writing_flag = 1
        if functional_group[5] == 1:
            if local_max_m_by_z[i] == 91:
                f.write("C7H7")
                f.write(" ")
                manual_writing_flag = 1
            if local_max_m_by_z[i] == 77:
                f.write("C6H5")
                f.write(" ")
                manual_writing_flag = 1
            if local_max_m_by_z[i] == 65:
                f.write("C5H5")
                f.write(" ")
                manual_writing_flag = 1

        if len(elemental_composition_dict)==3:
            if "C" in elemental_composition_dict and "H" in elemental_composition_dict and "O" in elemental_composition_dict:
                if local_max_m_by_z[i] == 16:
                    f.write("O")
                    f.write(" ")
                    manual_writing_flag = 1

        if len(elemental_composition_dict)==2:
            if "C" in elemental_composition_dict and "O" in elemental_composition_dict:
                if local_max_m_by_z[i] == 16:
                    f.write("O")
                    f.write(" ")
                    manual_writing_flag = 1

        if len(elemental_composition_dict)==2:
            if "H" in elemental_composition_dict and "O" in elemental_composition_dict:
                if local_max_m_by_z[i] == 16:
                    f.write("O")
                    f.write(" ")
                    manual_writing_flag = 1

        if len(elemental_composition_dict)==4:
            if "H" in elemental_composition_dict and "O" in elemental_composition_dict and "N" in elemental_composition_dict and "C" in elemental_composition_dict:
                if local_max_m_by_z[i] == 30:
                    f.write("NO")
                    f.write(" ")
                    manual_writing_flag = 1
        f.close()
        if manual_writing_flag == 1:
            continue
        if local_max_m_by_z[i] <= M:
            if M%2==0:
                odd=False
            else:
                odd=True
            #3-14 units sway from the molecular ion:
            m_three_away=M-3     #20 (fourteen 6- three 17)
            m_fourteen_away=M-14
            if local_max_m_by_z[i]<m_fourteen_away or local_max_m_by_z[i]>m_three_away:
                coincombination(local_max_m_by_z[i], 0, list2, odd=False, nitro=False, M=M, cfmid_frag_mass=cfmid_frag_mass, cfmid_frag_ions=cfmid_frag_ions, elements=elements,coins=coins, counting_coins=counting_coins,valency_coins=valency_coins, DOU_M=DOU_M)
        else:
            f = open("demofile.txt", "a")
            M_2_intensity = new_intensity_list[i]
            M_intensity = new_intensity_list[i - 2]

            if M_intensity==0:
                ratio_intensity=0
            else:
                ratio_intensity = M_2_intensity / M_intensity
            # either isotopes
            if "Cl" in elemental_composition_dict:
                if local_max_m_by_z[i] == M + 4 and elemental_composition_dict['Cl'] >= 2:
                    f.write(formula)
                    f.write(" ")
                if local_max_m_by_z[i] == M + 2 and elemental_composition_dict['Cl'] == 1:
                    f.write(formula)
                    f.write(" ")
            elif local_max_m_by_z[i] == M + 2 and "S" in elemental_composition_dict:
                if 0.044 <= ratio_intensity <= 0.05:
                    f.write(formula)
                    f.write(" ")
            elif "Br" in elemental_composition_dict:
                if local_max_m_by_z[i] == M + 4 and elemental_composition_dict['Br'] >= 2:
                    f.write(formula)
                    f.write(" ")
                if local_max_m_by_z[i] == M + 2 and elemental_composition_dict['Br'] == 1:
                    f.write(formula)
                    f.write(" ")
            else:
                #C13 isotope
                if "C" in elemental_composition_dict and "H" in elemental_composition_dict and len(elemental_composition_dict)==2:
                    if local_max_m_by_z[i] == M + 1:
                        f.write(formula)
                        f.write(" ")
                # or add by a hydrogen
                elif "H" in elemental_composition_dict:
                    if local_max_m_by_z[i] == M + 1:
                        if elemental_composition_dict['H']:
                            tmp_dict = elemental_composition_dict.copy()
                            tmp_dict['H'] = tmp_dict['H'] + 1
                            tmp_formula = dict_to_formula(tmp_dict)
                            f.write(tmp_formula)
                            f.write(" ")
                    if local_max_m_by_z[i] == M + 2:
                        if elemental_composition_dict['H']:
                            tmp_dict = elemental_composition_dict.copy()
                            tmp_dict['H'] = tmp_dict['H'] + 2
                            tmp_formula = dict_to_formula(tmp_dict)
                            f.write(tmp_formula)
                            f.write(" ")

            f.close()


    returned_annotation_list = [""] * (int(max_M)+1)

    with open("demofile.txt") as f:
        lines = f.readlines()
    prev_line=lines[0]
    for line in lines:
        p=line.split()
        flag = 0
        prev_p = prev_line.split(",")
        prev_p = prev_p[0].split(" ")
        prev_formula = returned_annotation_list[int(prev_p[1])]  # prev_formula = prev_p[2]
        prev_formula_dict = count_elements(prev_formula)
        sim_prev = int(prev_p[1]) + 1
        sim_p = int(p[1])
        if len(p)==3:
            flag=2
        elif len(p)>3 or len(p)<3:
            if len(p)<3:
                if flag==0 and sim_prev==sim_p:
                    new_formula_dict=prev_formula_dict.copy()
                    if new_formula_dict.get('H'):
                        new_formula_dict['H'] += 1
                        flag=1
                    else:
                        if len(new_formula_dict)>1:
                            new_formula_dict['H']= 1
                            flag=1
            for annot in range(2,len(p)):
                new_formula_dict=count_elements(p[annot])
                if set(new_formula_dict.keys()) == set(prev_formula_dict.keys()) and sim_prev==sim_p and flag==0:
                    if new_formula_dict.get('H') and prev_formula_dict.get('H'):
                        flag=1
                        break
            if len(p)>3 and flag==0:
                flag=2
            if len(p)<3 and flag==0:
                current_sim_p=sim_p
                current_sim_prev=sim_prev
                for looper in range(4):
                    if flag == 0 and current_sim_prev == current_sim_p:
                        new_formula_dict = prev_formula_dict.copy()
                        if new_formula_dict.get('H'):
                            new_formula_dict['H'] = new_formula_dict['H'] + looper + 1
                            flag = 1
                            break
                    current_sim_prev = current_sim_prev + 1

        prev_line=line
        if flag == 0:
            returned_annotation_list[int(p[1])] = ""
        if flag == 1:
            returned_annotation_list[int(p[1])] = dict_to_formula(new_formula_dict)
        if flag==2:
            if len(p)==3:
                returned_annotation_list[int(p[1])] = p[2]
            if len(p)>3:
                returned_annotation_list[int(p[1])]= ""

    return returned_annotation_list


