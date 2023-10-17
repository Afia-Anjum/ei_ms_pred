m_string="Aldehyde"
os.chdir('D:\\NIST20\\MSSEARCH\\pythonProject\\hold_out_test\\'+m_string+'\\')
with open("Aldehyde_chemically_valid_group_for_pagtn.txt" ) as f:
    content_list = f.readlines()
content_list = [(x.strip()) for x in content_list]
#print(content_list)

smiles=[]
test_number=[]
MW=[]
for i in range(len(content_list)):
    p_index = content_list[i].split()
    test_number.append((p_index[0]))
    smiles.append((p_index[1]))


for i in range(len(smiles)):
    MW.append(Descriptors.ExactMolWt(Chem.MolFromSmiles(smiles[i])))
peaks=[]
for i in range(len(test_number)):
    with open(test_number[i]+".txt") as f1:
        content_list2 = f1.readlines()
    content_list2 = [(x.strip()) for x in content_list2]
    content_list2.pop(0)
    content_list2.pop(0)
    content_list2.pop(0)
    #print(content_list2)
    separator = '; '
    tmp_list=separator.join(content_list2)
    peaks.append(tmp_list)

c=pd.DataFrame(smiles)
c.to_csv("input_for_PAGTN/smiles.csv",index=False)
c=pd.DataFrame(peaks)
c.to_csv("input_for_PAGTN/peaks.csv",index=False)
c=pd.DataFrame(MW)
c.to_csv("input_for_PAGTN/MW.csv",index=False)