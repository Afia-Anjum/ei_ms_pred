"""
Created on Tue May 18 16:54:01 2021

@author: Asus
"""

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
# from chembl_structure_pipeline import checker
from ChEMBL_Structure_Pipeline.chembl_structure_pipeline import checker



def transform(smarts_transformation, base_molecule):
    try:
        rxn = AllChem.ReactionFromSmarts(smarts_transformation)
        return rxn.RunReactants((base_molecule,))
    except:
        return -1



def remove_duplicates(products):
    products_hash = {}

    for i in range(0, len(products)):
        smiles = Chem.MolToSmiles(products[i][0], isomericSmiles=True, kekuleSmiles=False)
        if smiles not in products_hash:
            rd_mol = Chem.MolFromSmiles(smiles)
            if rd_mol:
                mol_wt = Chem.Descriptors.ExactMolWt(rd_mol)
                if mol_wt <= 900:
                    products_hash[smiles] = products[i][0]
    return products_hash


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
            issues = checker.check_molblock(m2)
            if len(issues) > 0:
                if not issues[0][0] == 5 or not issues[0][0] == 6:
                    new_list.append(list1[i])
            else:
                new_list.append(list1[i])

    return new_list


def create_array_dicts(smiles, tms_coh, tms_soh, tms_poh, tms_thiol, tms_ketone1, tms_ketone2, tms_ketone3,
                       tms_aldehyde1, tms_aldehyde2, tms_aldehyde3, tms_amine1, tms_amine2):
    n = 12  # n is a big number
    num_tms = n
    hashlist = [dict() for x in range(num_tms + 1)]

    hashlist[0][smiles] = smiles

    # Attach phenyl group to someplace on the alkyl chain
    # print "adding TMS " + str(i)
    flag = 0
    for i in range(1, num_tms + 1):
        for key in hashlist[i - 1]:
            ps = transform(tms_coh, Chem.MolFromSmiles(key))
            if ps == -1:
                flag = 1
                break

            ps_hash_tmp = remove_duplicates(ps)
            hashlist[i].update(ps_hash_tmp)

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


def create_array_dicts2(smiles, tms_coh, tms_soh, tms_poh, tms_thiol, tms_ketone1, tms_ketone2, tms_ketone3,
                        tms_aldehyde1, tms_aldehyde2, tms_aldehyde3, tms_amine1, tms_amine2, methoxy_OH, ethoxy_OH,
                        methylation_amine1, methylation_amine2, ethylation_amine1, ethylation_amine2, methylation_soh,
                        ethylation_soh, methylation_poh, ethylation_poh, methylation_thiol, ethylation_thiol):
    n = 12  # n is a big number
    num_tms = n
    hashlist = [dict() for x in range(num_tms + 1)]

    # hashlist[0][smiles] = Chem.Kekulize(Chem.MolFromSmiles(smiles)) # set consisting of initial molecule
    hashlist[0][smiles] = smiles

    # Attach phenyl group to someplace on the alkyl chain
    # print "adding TMS " + str(i)
    flag = 0
    for i in range(1, num_tms + 1):
        print("add TMS " + str(i))
        for key in hashlist[i - 1]:
            # alcohol attached to carbon
            ps = transform(tms_coh, Chem.MolFromSmiles(key))

            if ps == -1:
                flag = 1
                break

            ps_hash_tmp = remove_duplicates(ps)
            hashlist[i].update(ps_hash_tmp)

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

            # methoxy
            ps = transform(methoxy_OH, Chem.MolFromSmiles(key))
            ps_hash_tmp = remove_duplicates(ps)
            hashlist[i].update(ps_hash_tmp)

            # ethoxy
            ps = transform(ethoxy_OH, Chem.MolFromSmiles(key))
            ps_hash_tmp = remove_duplicates(ps)
            hashlist[i].update(ps_hash_tmp)

            # methylation_amine1
            ps = transform(methylation_amine1, Chem.MolFromSmiles(key))
            ps_hash_tmp = remove_duplicates(ps)
            hashlist[i].update(ps_hash_tmp)

            # methylation_amine2
            ps = transform(methylation_amine2, Chem.MolFromSmiles(key))
            ps_hash_tmp = remove_duplicates(ps)
            hashlist[i].update(ps_hash_tmp)

            # ethylation_amine1
            ps = transform(ethylation_amine1, Chem.MolFromSmiles(key))
            ps_hash_tmp = remove_duplicates(ps)
            hashlist[i].update(ps_hash_tmp)

            # ethylation_amine2
            ps = transform(ethylation_amine2, Chem.MolFromSmiles(key))
            ps_hash_tmp = remove_duplicates(ps)
            hashlist[i].update(ps_hash_tmp)

            # methylation_soh
            ps = transform(methylation_soh, Chem.MolFromSmiles(key))
            ps_hash_tmp = remove_duplicates(ps)
            hashlist[i].update(ps_hash_tmp)

            # ethylation_soh
            ps = transform(ethylation_soh, Chem.MolFromSmiles(key))
            ps_hash_tmp = remove_duplicates(ps)
            hashlist[i].update(ps_hash_tmp)

            # methylation_poh
            ps = transform(methylation_poh, Chem.MolFromSmiles(key))
            ps_hash_tmp = remove_duplicates(ps)
            hashlist[i].update(ps_hash_tmp)

            # ethylation_poh
            ps = transform(ethylation_poh, Chem.MolFromSmiles(key))
            ps_hash_tmp = remove_duplicates(ps)
            hashlist[i].update(ps_hash_tmp)

            # methylation_thiol
            ps = transform(methylation_thiol, Chem.MolFromSmiles(key))
            ps_hash_tmp = remove_duplicates(ps)
            hashlist[i].update(ps_hash_tmp)

            # ethylation_thiol
            ps = transform(ethylation_thiol, Chem.MolFromSmiles(key))
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


def make_tbdms(smiles):
    tms_coh = "[O;X2;H1:1][#6:2]>>[O;X2;H0:1]([#6:2])[Si](C)(C)C(C)(C)C"  # will also work on COOH
    tms_soh = "[O;X2;H1:1][S:2]>>[O;X2;H0:1]([S:2])[Si](C)(C)C(C)(C)C"
    tms_poh = "[O;X2;H1:1][P:2]>>[O;X2;H0:1]([P:2])[Si](C)(C)C(C)(C)C"
    # For OH, only allow for OH group attached to C, S, or P. (H2O, for example, should not match).
    tms_thiol = "[S;X2;H1:1]>>[S;X2;H0:1][Si](C)(C)C(C)(C)C"
    # tms_ketone = "[O;X1;D1;H0:1]=[C:2]([C:3])[C;X4;H1,H2,H3:4]>>[O;X2;D1;H0:1]([C:2]([C:3])=[C;X3:4])[Si](C)(C)C"
    tms_ketone1 = "[O;X1;D1;H0:1]=[C:2]([C:3])[C;X4;H1:4]>>[O;X2;D1;H0:1]([C:2]([C:3])=[C;X3;H0:4])[Si](C)(C)C(C)(C)C"
    tms_ketone2 = "[O;X1;D1;H0:1]=[C:2]([C:3])[C;X4;H2:4]>>[O;X2;D1;H0:1]([C:2]([C:3])=[C;X3;H1:4])[Si](C)(C)C(C)(C)C"
    tms_ketone3 = "[O;X1;D1;H0:1]=[C:2]([C:3])[C;X4;H3:4]>>[O;X2;D1;H0:1]([C:2]([C:3])=[C;X3;H2:4])[Si](C)(C)C(C)(C)C"
    # enolizable ketone
    # requires that the alpha carbon be aliphatic, and not involved in any double or triple bonds before the reaction
    # http://www.ochempal.org/index.php/alphabetical/e-f/enolizable-ketone/
    # tms_aldehyde = "[O;X1;D1;H0:1]=[C;H1:2][C;X4;H1,H2,H3:3]>>[O;X2;D1;H0:1]([C;H1:2]=[C;X3:3])[Si](C)(C)C"
    tms_aldehyde1 = "[O;X1;D1;H0:1]=[C;H1:2][C;X4;H1:3]>>[O;X2;D1;H0:1]([C;H1:2]=[C;X3;H0:3])[Si](C)(C)C(C)(C)C"
    tms_aldehyde2 = "[O;X1;D1;H0:1]=[C;H1:2][C;X4;H2:3]>>[O;X2;D1;H0:1]([C;H1:2]=[C;X3;H1:3])[Si](C)(C)C(C)(C)C"
    tms_aldehyde3 = "[O;X1;D1;H0:1]=[C;H1:2][C;X4;H3:3]>>[O;X2;D1;H0:1]([C;H1:2]=[C;X3;H2:3])[Si](C)(C)C(C)(C)C"
    # enolizable aldehyde
    # requires that the alpha carbon be aliphatic, and not involved in any double or triple bonds before the reaction
    # http://www.ochempal.org/index.php/alphabetical/e-f/enolizable-aldehyde/

    tms_amine1 = "[#7;H2:1]>>[#7;H1:1][Si](C)(C)C(C)(C)C"
    tms_amine2 = "[#7;H1:1]>>[#7;H0:1][Si](C)(C)C(C)(C)C"

    dict_array1 = create_array_dicts(smiles, tms_coh, tms_soh, tms_poh, tms_thiol, tms_ketone1, tms_ketone2,
                                     tms_ketone3, tms_aldehyde1, tms_aldehyde2, tms_aldehyde3, tms_amine1, tms_amine2)

    return dict_array1


def make_tbdps(smiles):
    tms_coh = "[O;X2;H1:1][#6:2]>>[O;X2;H0:1]([#6:2])CC(C)(C)[Si](C1=CC=CC=C1)C1=CC=CC=C1"  # will also work on COOH
    tms_soh = "[O;X2;H1:1][S:2]>>[O;X2;H0:1]([S:2])CC(C)(C)[Si](C1=CC=CC=C1)C1=CC=CC=C1"
    tms_poh = "[O;X2;H1:1][P:2]>>[O;X2;H0:1]([P:2])CC(C)(C)[Si](C1=CC=CC=C1)C1=CC=CC=C1"
    # For OH, only allow for OH group attached to C, S, or P. (H2O, for example, should not match).
    tms_thiol = "[S;X2;H1:1]>>[S;X2;H0:1]CC(C)(C)[Si](C1=CC=CC=C1)C1=CC=CC=C1"
    # tms_ketone = "[O;X1;D1;H0:1]=[C:2]([C:3])[C;X4;H1,H2,H3:4]>>[O;X2;D1;H0:1]([C:2]([C:3])=[C;X3:4])[Si](C)(C)C"
    tms_ketone1 = "[O;X1;D1;H0:1]=[C:2]([C:3])[C;X4;H1:4]>>[O;X2;D1;H0:1]([C:2]([C:3])=[C;X3;H0:4])CC(C)(C)[Si](C1=CC=CC=C1)C1=CC=CC=C1"
    tms_ketone2 = "[O;X1;D1;H0:1]=[C:2]([C:3])[C;X4;H2:4]>>[O;X2;D1;H0:1]([C:2]([C:3])=[C;X3;H1:4])CC(C)(C)[Si](C1=CC=CC=C1)C1=CC=CC=C1"
    tms_ketone3 = "[O;X1;D1;H0:1]=[C:2]([C:3])[C;X4;H3:4]>>[O;X2;D1;H0:1]([C:2]([C:3])=[C;X3;H2:4])CC(C)(C)[Si](C1=CC=CC=C1)C1=CC=CC=C1"
    # enolizable ketone
    # requires that the alpha carbon be aliphatic, and not involved in any double or triple bonds before the reaction
    # http://www.ochempal.org/index.php/alphabetical/e-f/enolizable-ketone/
    # tms_aldehyde = "[O;X1;D1;H0:1]=[C;H1:2][C;X4;H1,H2,H3:3]>>[O;X2;D1;H0:1]([C;H1:2]=[C;X3:3])[Si](C)(C)C"
    tms_aldehyde1 = "[O;X1;D1;H0:1]=[C;H1:2][C;X4;H1:3]>>[O;X2;D1;H0:1]([C;H1:2]=[C;X3;H0:3])CC(C)(C)[Si](C1=CC=CC=C1)C1=CC=CC=C1"
    tms_aldehyde2 = "[O;X1;D1;H0:1]=[C;H1:2][C;X4;H2:3]>>[O;X2;D1;H0:1]([C;H1:2]=[C;X3;H1:3])CC(C)(C)[Si](C1=CC=CC=C1)C1=CC=CC=C1"
    tms_aldehyde3 = "[O;X1;D1;H0:1]=[C;H1:2][C;X4;H3:3]>>[O;X2;D1;H0:1]([C;H1:2]=[C;X3;H2:3])CC(C)(C)[Si](C1=CC=CC=C1)C1=CC=CC=C1"

    # enolizable aldehyde
    # requires that the alpha carbon be aliphatic, and not involved in any double or triple bonds before the reaction
    # http://www.ochempal.org/index.php/alphabetical/e-f/enolizable-aldehyde/
    # tms_amine1 = "[#7;H2:1]>>[#7;H1:1][Si](C1=CC=CC=C1)C1=CC=CC=C1"
    # tms_amine2 = "[#7;H1:1]>>[#7;H0:1][Si](C1=CC=CC=C1)C1=CC=CC=C1"

    tms_amine1 = "[#7;H2:1]>>CC(C)(C)[Si]([#7;H1:1])(C1=CC=CC=C1)C1=CC=CC=C1"
    tms_amine2 = "[#7;H1:1]>>CC(C)(C)[Si]([#7;H0:1])(C1=CC=CC=C1)C1=CC=CC=C1"

    dict_array1 = create_array_dicts(smiles, tms_coh, tms_soh, tms_poh, tms_thiol, tms_ketone1, tms_ketone2,
                                     tms_ketone3, tms_aldehyde1, tms_aldehyde2, tms_aldehyde3, tms_amine1, tms_amine2)

    return dict_array1


def make_tips(smiles):
    tms_coh = "[O;X2;H1:1][#6:2]>>[O;X2;H0:1]([#6:2])[Si]([#6;A](C)(C)C)([#6;A](C)(C)C)[#6;A](C)(C)C"
    tms_soh = "[O;X2;H1:1][S:2]>>[O;X2;H0:1]([S:2])[Si]([#6;A](C)(C)C)([#6;A](C)(C)C)[#6;A](C)(C)C"
    tms_poh = "[O;X2;H1:1][P:2]>>[O;X2;H0:1]([P:2])[Si]([#6;A](C)(C)C)([#6;A](C)(C)C)[#6;A](C)(C)C"
    # For OH, only allow for OH group attached to C, S, or P. (H2O, for example, should not match).
    tms_thiol = "[S;X2;H1:1]>>[S;X2;H0:1][Si]([#6;A](C)(C)C)([#6;A](C)(C)C)[#6;A](C)(C)C"
    # tms_ketone = "[O;X1;D1;H0:1]=[C:2]([C:3])[C;X4;H1,H2,H3:4]>>[O;X2;D1;H0:1]([C:2]([C:3])=[C;X3:4])[Si](C)(C)C"
    tms_ketone1 = "[O;X1;D1;H0:1]=[C:2]([C:3])[C;X4;H1:4]>>[O;X2;D1;H0:1]([C:2]([C:3])=[C;X3;H0:4])[Si]([#6;A](C)(C)C)([#6;A](C)(C)C)[#6;A](C)(C)C"
    tms_ketone2 = "[O;X1;D1;H0:1]=[C:2]([C:3])[C;X4;H2:4]>>[O;X2;D1;H0:1]([C:2]([C:3])=[C;X3;H1:4])[Si]([#6;A](C)(C)C)([#6;A](C)(C)C)[#6;A](C)(C)C"
    tms_ketone3 = "[O;X1;D1;H0:1]=[C:2]([C:3])[C;X4;H3:4]>>[O;X2;D1;H0:1]([C:2]([C:3])=[C;X3;H2:4])[Si]([#6;A](C)(C)C)([#6;A](C)(C)C)[#6;A](C)(C)C"
    # enolizable ketone
    # requires that the alpha carbon be aliphatic, and not involved in any double or triple bonds before the reaction
    # http://www.ochempal.org/index.php/alphabetical/e-f/enolizable-ketone/
    # tms_aldehyde = "[O;X1;D1;H0:1]=[C;H1:2][C;X4;H1,H2,H3:3]>>[O;X2;D1;H0:1]([C;H1:2]=[C;X3:3])[Si](C)(C)C"
    tms_aldehyde1 = "[O;X1;D1;H0:1]=[C;H1:2][C;X4;H1:3]>>[O;X2;D1;H0:1]([C;H1:2]=[C;X3;H0:3])[Si]([#6;A](C)(C)C)([#6;A](C)(C)C)[#6;A](C)(C)C"
    tms_aldehyde2 = "[O;X1;D1;H0:1]=[C;H1:2][C;X4;H2:3]>>[O;X2;D1;H0:1]([C;H1:2]=[C;X3;H1:3])[Si]([#6;A](C)(C)C)([#6;A](C)(C)C)[#6;A](C)(C)C"
    tms_aldehyde3 = "[O;X1;D1;H0:1]=[C;H1:2][C;X4;H3:3]>>[O;X2;D1;H0:1]([C;H1:2]=[C;X3;H2:3])[Si]([#6;A](C)(C)C)([#6;A](C)(C)C)[#6;A](C)(C)C"
    # enolizable aldehyde
    # requires that the alpha carbon be aliphatic, and not involved in any double or triple bonds before the reaction
    # http://www.ochempal.org/index.php/alphabetical/e-f/enolizable-aldehyde/
    tms_amine1 = "[#7;H2:1]>>[#7;H1:1][Si]([#6;A](C)(C)C)([#6;A](C)(C)C)[#6;A](C)(C)C"
    tms_amine2 = "[#7;H1:1]>>[#7;H0:1][Si]([#6;A](C)(C)C)([#6;A](C)(C)C)[#6;A](C)(C)C"

    dict_array1 = create_array_dicts(smiles, tms_coh, tms_soh, tms_poh, tms_thiol, tms_ketone1, tms_ketone2,
                                     tms_ketone3, tms_aldehyde1, tms_aldehyde2, tms_aldehyde3, tms_amine1, tms_amine2)

    return dict_array1


def make_tes(smiles):
    tms_coh = "[O;X2;H1:1][#6:2]>>[O;X2;H0:1]([#6:2])[Si](CC)(CC)CC"  # will also work on COOH
    tms_soh = "[O;X2;H1:1][S:2]>>[O;X2;H0:1]([S:2])[Si](CC)(CC)CC"
    tms_poh = "[O;X2;H1:1][P:2]>>[O;X2;H0:1]([P:2])[Si](CC)(CC)CC"
    # For OH, only allow for OH group attached to C, S, or P. (H2O, for example, should not match).
    tms_thiol = "[S;X2;H1:1]>>[S;X2;H0:1][Si](CC)(CC)CC"
    # tms_ketone = "[O;X1;D1;H0:1]=[C:2]([C:3])[C;X4;H1,H2,H3:4]>>[O;X2;D1;H0:1]([C:2]([C:3])=[C;X3:4])[Si](CC)(CC)CC"
    tms_ketone1 = "[O;X1;D1;H0:1]=[C:2]([C:3])[C;X4;H1:4]>>[O;X2;D1;H0:1]([C:2]([C:3])=[C;X3;H0:4])[Si](CC)(CC)CC"
    tms_ketone2 = "[O;X1;D1;H0:1]=[C:2]([C:3])[C;X4;H2:4]>>[O;X2;D1;H0:1]([C:2]([C:3])=[C;X3;H1:4])[Si](CC)(CC)CC"
    tms_ketone3 = "[O;X1;D1;H0:1]=[C:2]([C:3])[C;X4;H3:4]>>[O;X2;D1;H0:1]([C:2]([C:3])=[C;X3;H2:4])[Si](CC)(CC)CC"
    # enolizable ketone
    # requires that the alpha carbon be aliphatic, and not involved in any double or triple bonds before the reaction
    # http://www.ochempal.org/index.php/alphabetical/e-f/enolizable-ketone/
    # tms_aldehyde = "[O;X1;D1;H0:1]=[C;H1:2][C;X4;H1,H2,H3:3]>>[O;X2;D1;H0:1]([C;H1:2]=[C;X3:3])[Si](CC)(CC)CC"
    tms_aldehyde1 = "[O;X1;D1;H0:1]=[C;H1:2][C;X4;H1:3]>>[O;X2;D1;H0:1]([C;H1:2]=[C;X3;H0:3])[Si](CC)(CC)CC"
    tms_aldehyde2 = "[O;X1;D1;H0:1]=[C;H1:2][C;X4;H2:3]>>[O;X2;D1;H0:1]([C;H1:2]=[C;X3;H1:3])[Si](CC)(CC)CC"
    tms_aldehyde3 = "[O;X1;D1;H0:1]=[C;H1:2][C;X4;H3:3]>>[O;X2;D1;H0:1]([C;H1:2]=[C;X3;H2:3])[Si](CC)(CC)CC"
    # enolizable aldehyde
    # requires that the alpha carbon be aliphatic, and not involved in any double or triple bonds before the reaction
    # http://www.ochempal.org/index.php/alphabetical/e-f/enolizable-aldehyde/
    tms_amine1 = "[#7;H2:1]>>[#7;H1:1][Si](CC)(CC)CC"
    tms_amine2 = "[#7;H1:1]>>[#7;H0:1][Si](CC)(CC)CC"

    dict_array1 = create_array_dicts(smiles, tms_coh, tms_soh, tms_poh, tms_thiol, tms_ketone1, tms_ketone2,
                                     tms_ketone3, tms_aldehyde1, tms_aldehyde2, tms_aldehyde3, tms_amine1, tms_amine2)

    return dict_array1


def make_tfa(smiles):
    tms_coh = "[O;X2;H1:1][#6:2]>>C-[#8]-C(=O)C(F)(F)F"
    tms_soh = "[#8;H1X2:1]-[#16]>>FC(F)(F)[#6](=O)-[#8]-[#16]"
    tms_poh = "[#8;H1X2:1]-[#15]>>FC(F)(F)[#6](=O)-[#8]-[#15]"
    # For OH, only allow for OH group attached to C, S, or P. (H2O, for example, should not match).
    tms_thiol = "[#16;A;H1X2:1]>>FC(F)(F)[#6](-[#16;H0X2:1])=O"
    # tms_ketone = "[O;X1;D1;H0:1]=[C:2]([C:3])[C;X4;H1,H2,H3:4]>>[O;X2;D1;H0:1]([C:2]([C:3])=[C;X3:4])[Si](C)(C)C"

    # tms_ketone1 = "[O;X1;D1;H0:1]=[C:2]([C:3])[C;X4;H1:4]>>[O;X2;D1;H0:1]([C:2]([C:3])=[C;X3;H0:4])[Si](C)(C)C"
    tms_ketone1 = "[#6;A;H1X4:4][#6:2]([#6;A:3])=[O;H0X1D1:1]>>[#6;A:3][#6:2](=[#6;A;H0X3:4])-[#8;H0X2D1:1]-[#6](=O)C(F)(F)F"
    tms_ketone2 = "[#6;A;H2X4:4][#6:2]([#6;A:3])=[O;H0X1D1:1]>>[#6;A:3][#6:2](=[#6;A;H1X3:4])-[#8;H0X2D1:1]-[#6](=O)C(F)(F)F"
    tms_ketone3 = "[#6;A:3][#6:2]([#6;A;H3X4:4])=[O;H0X1D1:1]>>[#6;A:3][#6:2](=[#6;A;H2X3:4])-[#8;H0X2D1:1]-[#6](=O)C(F)(F)F"
    # enolizable ketone
    # requires that the alpha carbon be aliphatic, and not involved in any double or triple bonds before the reaction
    # http://www.ochempal.org/index.php/alphabetical/e-f/enolizable-ketone/
    # tms_aldehyde = "[O;X1;D1;H0:1]=[C;H1:2][C;X4;H1,H2,H3:3]>>[O;X2;D1;H0:1]([C;H1:2]=[C;X3:3])[Si](C)(C)C"
    tms_aldehyde1 = "[#6;A;H1X4:3][#6H1:2]=[O;H0X1D1:1]>>FC(F)(F)[#6](=O)-[#8;H0X2D1:1]-[#6H1:2]=[#6;A;H0X3:3]"
    tms_aldehyde2 = "[#6;A;H2X4:3][#6H1:2]=[O;H0X1D1:1]>>FC(F)(F)[#6](=O)-[#8;H0X2D1:1]-[#6H1:2]=[#6;A;H1X3:3]"
    tms_aldehyde3 = "[#6;A;H3X4:3][#6H1:2]=[O;H0X1D1:1]>>FC(F)(F)[#6](=O)-[#8;H0X2D1:1]-[#6H1:2]=[#6;A;H2X3:3]"
    # enolizable aldehyde
    # requires that the alpha carbon be aliphatic, and not involved in any double or triple bonds before the reaction
    # http://www.ochempal.org/index.php/alphabetical/e-f/enolizable-aldehyde/
    tms_amine1 = "[#7H2:1]>>[#7H1:1]-[#6](=O)C(F)(F)F"
    tms_amine2 = "[#7H1:1]>>[#7H0:1]-[#6](=O)C(F)(F)F"

    dict_array1 = create_array_dicts(smiles, tms_coh, tms_soh, tms_poh, tms_thiol, tms_ketone1, tms_ketone2,
                                     tms_ketone3, tms_aldehyde1, tms_aldehyde2, tms_aldehyde3, tms_amine1, tms_amine2)

    return dict_array1


def make_oximation(smiles):
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

    methoxy_OH = "[#6:2]-[#8;H1X2:1]>>[#6:2]-[#8;H0X2:1][#6;A]"
    ethoxy_OH = "[#6:2]-[#8;H1X2:1]>>[#6:2]-[#8;H0X2:1][#6;A][#6;A]"

    methylation_amine1 = "[#7;H2:1]>>[#7;H1:1][#6;A]"
    ethylation_amine1 = "[#7;H2:1]>>[#7;H1:1][#6;A][#6;A]"

    methylation_amine2 = "[#7;H1:1]>>[#7;H0:1][#6;A]"
    ethylation_amine2 = "[#7;H1:1]>>[#7;H0:1][#6;A][#6;A]"

    methylation_soh = "[O;X2;H1:1][S:2]>>[O;X2;H0:1]([S:2])[#6;A]"
    ethylation_soh = "[O;X2;H1:1][S:2]>>[O;X2;H0:1]([S:2])[#6;A][#6;A]"

    methylation_poh = "[O;X2;H1:1][P:2]>>[O;X2;H0:1]([P:2])[#6;A]"
    ethylation_poh = "[O;X2;H1:1][P:2]>>[O;X2;H0:1]([P:2])[#6;A][#6;A]"

    methylation_thiol = "[S;X2;H1:1]>>[S;X2;H0:1][#6;A]"
    ethylation_thiol = "[S;X2;H1:1]>>[S;X2;H0:1][#6;A][#6;A]"

    dict_array1 = create_array_dicts2(smiles, tms_coh, tms_soh, tms_poh, tms_thiol, tms_ketone1, tms_ketone2,
                                      tms_ketone3, tms_aldehyde1, tms_aldehyde2, tms_aldehyde3, tms_amine1, tms_amine2,
                                      methoxy_OH, ethoxy_OH, methylation_amine1, methylation_amine2, ethylation_amine1,
                                      ethylation_amine2, methylation_soh, ethylation_soh, methylation_poh,
                                      ethylation_poh, methylation_thiol, ethylation_thiol)

    return dict_array1


start_time = time.time()



df = pd.read_csv('DrugBank_for_afia.csv', sep=',', error_bad_lines=False, index_col=False, dtype='unicode')
smilesdf = df['moldb_smiles']
drug_id_df = df['drugbank_id']
smileslist1 = smilesdf.values.tolist()
drug_id_list = drug_id_df.values.tolist()

finallist = []


list3 = smileslist1

end_time = time.time()
diff_time = end_time - start_time
print("  done in %f seconds" % diff_time)

for i in range(0, len(list3)):
    if i % 1000 == 0:
        print(str(i) + "processings are done")

    start_time = time.time()

    dict_array1 = make_tms(list3[i])
    dict_array2 = make_tbdms(list3[i])

    dict_array = []

    dict_array = dict_array + dict_array1
    dict_array = dict_array + dict_array2

    l = remove_dup_list(dict_array)

    l = check_issue(l)
    finallist.append(l)
    end_time = time.time()
    diff_time = end_time - start_time
    # print("  done in %f seconds" % diff_time)

print(len(finallist))


new_hmdb_id = []
new_smiles_list = []
new_names_list = []
string = ""
cnt = 0
for i in range(len(list3)):
    der_length = len(finallist[i])
    der_lists = finallist[i]
    string2 = 1
    for j in range(der_length):
        new_smiles_list.append(der_lists[j])
        if '[Si](C)(C)C(C)(C)C' in der_lists[j] or 'CC(C)(C)[Si](C)(C)' in der_lists[j]:
            string = "TBDMS"
        elif '[Si](C)(C)C' in der_lists[j] or 'C[Si](C)(C)' in der_lists[j]:
            string = "TMS"
        count_Si = der_lists[j].count("Si")

        if len(new_hmdb_id) > 0:
            splitting = new_hmdb_id[cnt - 1].split("_")
            if not splitting[0] == drug_id_list[i]:
                string2 = 1

            else:
                if string == splitting[1]:
                    if str(count_Si) == splitting[2]:
                        string2 = int(splitting[3]) + 1
                    else:
                        string2 = 1
                else:
                    string2 = 1
        # string2
        new_hmdb_id.append(drug_id_list[i] + "_" + string + "_" + str(count_Si) + "_" + str(string2))
        new_names_list.append(drug_id_list[i] + "," + str(count_Si) + string + ",isomer#" + str(string2))
        print(drug_id_list[i] + "_" + string + "_" + str(count_Si) + "_" + str(string2))
        cnt = cnt + 1

pd.DataFrame(new_hmdb_id).to_csv("new_drug_id.csv", index=False)
pd.DataFrame(new_names_list).to_csv("new_drug_names.csv", index=False)
pd.DataFrame(new_smiles_list).to_csv("new_drug_smiles.csv", index=False)