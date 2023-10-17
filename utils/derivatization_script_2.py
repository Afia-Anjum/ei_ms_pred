import numba
from numba import jit, cuda
import pandas as pd
import os, sys, re
import time
from optparse import OptionParser
from rdkit import Chem
from rdkit.Chem import AllChem
import csv
import rdkit.Chem.Descriptors
#from chembl_structure_pipeline import checker


def transform(smarts_transformation, base_molecule):
    try:
        rxn = AllChem.ReactionFromSmarts(smarts_transformation)
        return rxn.RunReactants((base_molecule,))
    except:
        return -1

def remove_dup_list(list1):
    list2 = []

    for i in range(len(list1)):
        if list1[i] not in list2:
            list2.append(list1[i])

    return list2


def check_issue(list1):
    new_list = []
    for i in range(len(list1)):
        m = Chem.MolFromSmiles(list1[i])
        if m is not None:
            m2 = Chem.MolToMolBlock(m)
            #issues = checker.check_molblock(m2)
            issues = check_molblock(m2)
            # print(issues)
            if len(issues) > 0:
                if not issues[0][0] == 5 or not issues[0][0] == 6:
                    new_list.append(list1[i])
            else:
                new_list.append(list1[i])

    return new_list


def remove_duplicates(products):
    products_hash = {}

    for i in range(0, len(products)):
        smiles = Chem.MolToSmiles(products[i][0], isomericSmiles=True, kekuleSmiles=False)
        if smiles not in products_hash:
            rd_mol = Chem.MolFromSmiles(smiles)
            mol_wt = Chem.Descriptors.ExactMolWt(rd_mol)
            # print(mol_wt)
            if mol_wt <= 900:
                products_hash[smiles] = products[i][0]
    return products_hash

def create_array_dicts(smiles, tms_coh, tms_soh, tms_poh, tms_thiol, tms_ketone1, tms_ketone2, tms_ketone3,
                       tms_aldehyde1, tms_aldehyde2, tms_aldehyde3, tms_amine1, tms_amine2):
    n = 12  # n is a big number

    num_tms = n
    hashlist = [dict() for x in range(num_tms + 1)]

    # hashlist[0][smiles] = Chem.Kekulize(Chem.MolFromSmiles(smiles)) # set consisting of initial molecule
    hashlist[0][smiles] = smiles

    # Attach phenyl group to someplace on the alkyl chain
    # print "adding TMS " + str(i)
    flag = 0
    for i in range(1, num_tms + 1):
        # print(hashlist)
        # print("Printing i:")
        # print(i)
        # print("add TMS " + str(i))
        for key in hashlist[i - 1]:
            # alcohol attached to carbon
            # print("printing key:")
            # print(key)
            ps = transform(tms_coh, Chem.MolFromSmiles(key))
            # print("printing ps:")
            #print(ps)
            if ps == -1:
                flag = 1
                break

            ps_hash_tmp = remove_duplicates(ps)
            hashlist[i].update(ps_hash_tmp)

            #print(hashlist)

            # OH group attached to S (e.g. sulphate group)_
            ps = transform(tms_soh, Chem.MolFromSmiles(key))
            ps_hash_tmp = remove_duplicates(ps)
            hashlist[i].update(ps_hash_tmp)

            # OH group attached to P (e.g. phosphate group)_
            ps = transform(tms_poh, Chem.MolFromSmiles(key))
            ps_hash_tmp = remove_duplicates(ps)
            hashlist[i].update(ps_hash_tmp)

            # thiol
            ps = transform(tms_thiol, Chem.MolFromSmiles(key))
            ps_hash_tmp = remove_duplicates(ps)
            hashlist[i].update(ps_hash_tmp)

            # enolizable ketone
            ps = transform(tms_ketone1, Chem.MolFromSmiles(key))
            ps_hash_tmp = remove_duplicates(ps)
            hashlist[i].update(ps_hash_tmp)
            ps = transform(tms_ketone2, Chem.MolFromSmiles(key))
            ps_hash_tmp = remove_duplicates(ps)
            hashlist[i].update(ps_hash_tmp)
            ps = transform(tms_ketone3, Chem.MolFromSmiles(key))
            ps_hash_tmp = remove_duplicates(ps)
            hashlist[i].update(ps_hash_tmp)

            # enolizable aldehyde
            ps = transform(tms_aldehyde1, Chem.MolFromSmiles(key))
            ps_hash_tmp = remove_duplicates(ps)
            hashlist[i].update(ps_hash_tmp)
            ps = transform(tms_aldehyde2, Chem.MolFromSmiles(key))
            ps_hash_tmp = remove_duplicates(ps)
            hashlist[i].update(ps_hash_tmp)
            ps = transform(tms_aldehyde3, Chem.MolFromSmiles(key))
            ps_hash_tmp = remove_duplicates(ps)
            hashlist[i].update(ps_hash_tmp)

            # amine 1
            ps = transform(tms_amine1, Chem.MolFromSmiles(key))
            ps_hash_tmp = remove_duplicates(ps)
            hashlist[i].update(ps_hash_tmp)

            # amine 2
            ps = transform(tms_amine2, Chem.MolFromSmiles(key))
            ps_hash_tmp = remove_duplicates(ps)
            hashlist[i].update(ps_hash_tmp)

        if flag == 1:
            break
        if len(hashlist[i]) == 0:
            break

    # if flag==1:
    #    return "Cannot process"
    # Output structures
    # output = open(outfile, 'w')
    count = 0
    smiles_kekulized_all = []

    for i in range(1, num_tms + 1):
        for key in hashlist[i]:
            # Format SMILES to show explicit bonds in aromatic rings (Kekule form)
            mol = Chem.MolFromSmiles(key)
            Chem.Kekulize(mol)
            smiles_kekulized = Chem.MolToSmiles(mol, isomericSmiles=True, kekuleSmiles=True)
            smiles_kekulized_all.append(smiles_kekulized)
            count += 1

    return smiles_kekulized_all


#def make_tms(smiles,n):
def make_tms(smiles):
    tms_coh = "[O;X2;H1:1][#6:2]>>[O;X2;H0:1]([#6:2])[Si](C)(C)C"  # will also work on COOH
    tms_soh = "[O;X2;H1:1][S:2]>>[O;X2;H0:1]([S:2])[Si](C)(C)C"
    tms_poh = "[O;X2;H1:1][P:2]>>[O;X2;H0:1]([P:2])[Si](C)(C)C"
    # For OH, only allow for OH group attached to C, S, or P. (H2O, for example, should not match).
    tms_thiol = "[S;X2;H1:1]>>[S;X2;H0:1][Si](C)(C)C"
    # tms_ketone = "[O;X1;D1;H0:1]=[C:2]([C:3])[C;X4;H1,H2,H3:4]>>[O;X2;D1;H0:1]([C:2]([C:3])=[C;X3:4])[Si](C)(C)C"
    tms_ketone1 = "[O;X1;D1;H0:1]=[C:2]([C:3])[C;X4;H1:4]>>[O;X2;D1;H0:1]([C:2]([C:3])=[C;X3;H0:4])[Si](C)(C)C"
    tms_ketone2 = "[O;X1;D1;H0:1]=[C:2]([C:3])[C;X4;H2:4]>>[O;X2;D1;H0:1]([C:2]([C:3])=[C;X3;H1:4])[Si](C)(C)C"
    tms_ketone3 = "[O;X1;D1;H0:1]=[C:2]([C:3])[C;X4;H3:4]>>[O;X2;D1;H0:1]([C:2]([C:3])=[C;X3;H2:4])[Si](C)(C)C"
    # enolizable ketone
    # requires that the alpha carbon be aliphatic, and not involved in any double or triple bonds before the reaction
    # http://www.ochempal.org/index.php/alphabetical/e-f/enolizable-ketone/
    # tms_aldehyde = "[O;X1;D1;H0:1]=[C;H1:2][C;X4;H1,H2,H3:3]>>[O;X2;D1;H0:1]([C;H1:2]=[C;X3:3])[Si](C)(C)C"
    tms_aldehyde1 = "[O;X1;D1;H0:1]=[C;H1:2][C;X4;H1:3]>>[O;X2;D1;H0:1]([C;H1:2]=[C;X3;H0:3])[Si](C)(C)C"
    tms_aldehyde2 = "[O;X1;D1;H0:1]=[C;H1:2][C;X4;H2:3]>>[O;X2;D1;H0:1]([C;H1:2]=[C;X3;H1:3])[Si](C)(C)C"
    tms_aldehyde3 = "[O;X1;D1;H0:1]=[C;H1:2][C;X4;H3:3]>>[O;X2;D1;H0:1]([C;H1:2]=[C;X3;H2:3])[Si](C)(C)C"
    # enolizable aldehyde
    # requires that the alpha carbon be aliphatic, and not involved in any double or triple bonds before the reaction
    # http://www.ochempal.org/index.php/alphabetical/e-f/enolizable-aldehyde/
    tms_amine1 = "[#7;H2:1]>>[#7;H1:1][Si](C)(C)C"
    tms_amine2 = "[#7;H1:1]>>[#7;H0:1][Si](C)(C)C"

    dict_array1 = create_array_dicts(smiles, tms_coh, tms_soh, tms_poh, tms_thiol, tms_ketone1, tms_ketone2,
                                     tms_ketone3, tms_aldehyde1, tms_aldehyde2, tms_aldehyde3, tms_amine1, tms_amine2)

    return dict_array1

import os
os.chdir('D:\\ALBERTA\\MimeDB')
m = pd.read_csv("MimeDB.csv", sep=',', error_bad_lines=False, index_col=False, dtype='unicode')
mime_id=m['id'].values.tolist()
#number_tms=m['TMS_number'].values.tolist()
mime_smiles=m['SMILES'].values.tolist()

#number_tms=[2]
#m_smiles=['C(C(=O)O)C(=O)O']

finallist=[]

import random

for i in range(len(mime_smiles)):

    start_time = time.time()
    #n=int(number_tms[i])
    #dict_array1 = make_tms(m_smiles[i],n)
    dict_array1 = make_tms(mime_smiles[i])
    dict_array = []
    dict_array = dict_array + dict_array1
    l = remove_dup_list(dict_array)
    l = check_issue(l)

    #print(l)
    # pop_list=[]
    # for i in range(len(l)):
    #     count_Si = l[i].count("Si")
    #     if count_Si == n:
    #         pop_list.append(l[i])

    # if len(pop_list)==0:
    #     finallist.append("No derivative")
    #     continue
    # pl=random.choice(pop_list)
    # print(pl)
    finallist.append(l)
    end_time = time.time()
    diff_time = end_time - start_time
    print("  done in %f seconds" % diff_time)

#######DERIVATIZATION SCRIPT ENDS HERE ###########

s=pd.DataFrame(finallist)
s.to_csv("mimedb_tms.csv",index=False)

import os
os.chdir('D:\\ALBERTA\\MimeDB')
m = pd.read_csv("MimeDB.csv", sep=',', error_bad_lines=False, index_col=False, dtype='unicode')
mime_id=m['id'].values.tolist()
#number_tms=m['TMS_number'].values.tolist()
mime_smiles=m['SMILES'].values.tolist()

m = pd.read_csv("MimeDB.csv", sep=',', error_bad_lines=False, index_col=False, dtype='unicode')