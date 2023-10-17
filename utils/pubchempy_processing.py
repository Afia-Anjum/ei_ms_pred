import pubchempy as pcp

small_molecules = pd.read_csv('small_molecules_set.csv', sep=',', error_bad_lines=False, index_col=False, dtype='unicode')
small_molecules_inchi_list = small_molecules['inchi'].values.tolist()

smiles_tms=[]

for i in range(len(small_molecules_inchi_list)):
    smiles_tms.append(Chem.MolToSmiles(Chem.MolFromInchi(small_molecules_inchi_list[i])))
for i in range(len(small_molecules_inchi_list)):

    #if pcp.get_compounds(name_m_tms[i], 'name'):
    #    for compound in pcp.get_compounds(name_m_tms[i], 'name'):
    if pcp.get_compounds(small_molecules_inchi_list[i], 'inchi'):
        for compound in pcp.get_compounds(small_molecules_inchi_list[i], 'inchi'):
            # print(compound.isomeric_smiles)
            # print(compound.iupac_name)
            # print(compound.inchikey)
            if (compound.cid):
                smiles_tms.append(compound.isomeric_smiles)
                # print("smiles found")
                #print(compound.isomeric_smiles)
                # print(compound.cid)
                # print(compound.isomeric_smiles)
                # print(compound.iupac_name)
            else:
                # print("smiles not found1")
                smiles_tms.append("")
            break
    else:
        # print("smiles not found2")
        # print("name not found")
        smiles_tms.append("")