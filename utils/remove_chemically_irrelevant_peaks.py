####CODE FOR REMOVING CHEMICALLY IRRELEVANT PEAKS: for aldehydes#######
m_string="Aldehyde"
os.chdir('D:\\NIST20\\MSSEARCH\\pythonProject\\hold_out_test\\'+m_string+'\\')
with open(m_string+"_test_heldout.txt" ) as f:
    content_list = f.readlines()
content_list = [(x.strip()) for x in content_list]
#content_list.pop(0)

smiles=[]
test_number=[]
m=0
n=0
for i in range(len(content_list)):
    #print(i)
    p_index = content_list[i].split()
    with open(p_index[0]+".txt") as f1:
        content_list2 = f1.readlines()
    content_list2 = [(x.strip()) for x in content_list2]
    content_list2.pop(0)
    content_list2.pop(0)
    content_list2.pop(0)
    p2_mass=[]
    p2_intensity=[]
    for l in range(len(content_list2)):
        p2_index=content_list2[l].split()
        p2_mass.append(int(p2_index[0]))
        p2_intensity.append(p2_index[1])


    Bold_M=int(Descriptors.ExactMolWt(Chem.MolFromSmiles(p_index[1])))
    print(Bold_M)
    if 44 in p2_mass and 29 in p2_mass and Bold_M-18 in p2_mass and Bold_M-28 in p2_mass and Bold_M-44 in p2_mass:
        smiles.append(p_index[1])
        test_number.append(p_index[0])


#print(m)
#print(n)
f = open("input_example_"+m_string+"_new_held_out_test_chemically_valid_group_for_cfmid.txt", "w")
for i in range(len(test_number)):
#for i in range(5):
    #print(smiles[i])
    #f = open("Test"+str(i+1)+".txt", "a")
    f.write(test_number[i]+" ")
    f.write(smiles[i]+" 1"+"\n")

f = open("input_example_"+m_string+"_new_held_out_test_chemically_valid_group_for_pagtn.txt", "w")
for i in range(len(test_number)):
#for i in range(5):
    #print(smiles[i])
    #f = open("Test"+str(i+1)+".txt", "a")
    f.write(test_number[i]+" ")
    f.write(smiles[i]+"\n")


####For Silylated compounds#######
os.chdir('D:\\NIST20\\MSSEARCH\\pythonProject\\hold_out_test\\Si\\')
with open("Si_test_heldout.txt" ) as f:
    content_list = f.readlines()
content_list = [(x.strip()) for x in content_list]
#content_list.pop(0)

smiles=[]
test_number=[]
m=0
n=0
for i in range(len(content_list)):
    print(i)
    p_index = content_list[i].split()
    #mass.append(int(float(p_index[0])))
    cnt = p_index[1].count("Si")
    #if cnt>4:
    #    print(p_index[0])
    if cnt <= 4:
        if '[Si](C)(C)C(C)(C)C' in p_index[1] or 'CC(C)(C)[Si](C)(C)' in p_index[1] or '[Si](C)(C)C' in p_index[1] or 'C[Si](C)(C)' in p_index[1]:
            with open(p_index[0]+".txt") as f1:
                content_list2 = f1.readlines()
            content_list2 = [(x.strip()) for x in content_list2]
            content_list2.pop(0)
            content_list2.pop(0)
            content_list2.pop(0)
            p2_mass=[]
            p2_intensity=[]
            #print(content_list2)
            for l in range(len(content_list2)):
                p2_index=content_list2[l].split()
                p2_mass.append(int(p2_index[0]))
                p2_intensity.append(p2_index[1])
            if cnt==1:
                if '[Si](C)(C)C(C)(C)C' in p_index[1] or 'CC(C)(C)[Si](C)(C)' in p_index[1]:
                    if 73 in p2_mass and (112 in p2_mass or 115 in p2_mass):
                        smiles.append(p_index[1])
                        test_number.append(p_index[0])
                elif '[Si](C)(C)C' in p_index[1] or 'C[Si](C)(C)' in p_index[1]:
                    if 73 in p2_mass:
                        smiles.append(p_index[1])
                        test_number.append(p_index[0])
            if cnt==2:
                if '[Si](C)(C)C(C)(C)C' in p_index[1] or 'CC(C)(C)[Si](C)(C)' in p_index[1]:
                    if 73 in p2_mass and (112 in p2_mass or 115 in p2_mass) and (144 in p2_mass or 147 in p2_mass):
                        smiles.append(p_index[1])
                        test_number.append(p_index[0])
                elif '[Si](C)(C)C' in p_index[1] or 'C[Si](C)(C)' in p_index[1]:
                    if 73 in p2_mass and (144 in p2_mass or 147 in p2_mass):
                        smiles.append(p_index[1])
                        test_number.append(p_index[0])
            if cnt==3:
                if '[Si](C)(C)C(C)(C)C' in p_index[1] or 'CC(C)(C)[Si](C)(C)' in p_index[1]:
                    continue
                elif '[Si](C)(C)C' in p_index[1] or 'C[Si](C)(C)' in p_index[1]:
                    if 73 in p2_mass and (144 in p2_mass or 147 in p2_mass) and (217 in p2_mass or 218 in p2_mass or 219 in p2_mass or 220 in p2_mass or 221 in p2_mass):
                        smiles.append(p_index[1])
                        test_number.append(p_index[0])

#print(m)
#print(n)
f = open("input_example_Si_new_held_out_test_chemically_valid_group_for_cfmid.txt", "w")
for i in range(len(test_number)):
#for i in range(5):
    #print(smiles[i])
    #f = open("Test"+str(i+1)+".txt", "a")
    f.write(test_number[i]+" ")
    f.write(smiles[i]+" 1"+"\n")

f = open("input_example_Si_new_held_out_test_chemically_valid_group_for_pagtn.txt", "w")
for i in range(len(test_number)):
#for i in range(5):
    #print(smiles[i])
    #f = open("Test"+str(i+1)+".txt", "a")
    f.write(test_number[i]+" ")
    f.write(smiles[i]+"\n")
