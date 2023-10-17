m_string="Ester"
os.chdir('D:\\NIST20\\MSSEARCH\\pythonProject\\hold_out_test\\'+m_string+'\\')

with open(m_string+"_test_heldout.txt" ) as f:
    content_list = f.readlines()
content_list = [(x.strip()) for x in content_list]
test_number=[]
smiles=[]
for i in range(len(content_list)):
   p_index = content_list[i].split()
   test_number.append(p_index[0])
   smiles.append(p_index[1])

f = open("input_example_"+m_string+"_new_held_out_test_chemically_valid_group_for_cfmid.txt", "w")
for i in range(len(test_number)):
#for i in range(5):
    #print(smiles[i])
    #f = open("Test"+str(i+1)+".txt", "a")
    f.write(test_number[i]+" ")
    f.write(smiles[i]+" 1"+"\n")
