
Car_Ni_orig = pd.read_csv('Car_Ni_orig.csv', sep=',', error_bad_lines=False, index_col=False, dtype='unicode')
result_Car_Ni = pd.merge(hold_out, Car_Ni_orig, on="smiles",how="inner")
result_Car_Ni=result_Car_Ni[result_Car_Ni['group']=='Carbonyl_nitrogen']
result_Car_Ni.to_csv("result_Car_Ni.csv",index=False)

os.chdir('D:\\NIST20\\MSSEARCH\\pythonProject\\')
raw = pd.read_csv('result_MW_less_orig.csv', sep=',', error_bad_lines=False, index_col=False, dtype='unicode')
peaks=raw['peaks'].tolist()
name=raw['name'].tolist()
num_peaks=raw['Num_peaks'].tolist()
smiles=raw['smiles'].tolist()
os.chdir('D:\\NIST20\\MSSEARCH\\pythonProject\\hold_out_test\\MW_less\\')
string_group="_MW_less_"
f = open("MW_less_test_heldout.txt", "w")
for i in range(len(smiles)):
#for i in range(5):
    #print(smiles[i])
    #f = open("Test"+str(i+1)+".txt", "a")
    f.write("Hold_out_Test"+string_group+str(i+1)+" ")
    f.write(smiles[i]+"\n")
f.close()
#os.chdir('D:\\NIST20\\MSSEARCH\\pythonProject\\EI-MS_train_text_files\\Si\\Si_test_100\\')
for i in range(len(smiles)):
    #print(smiles[i])
    f = open("Hold_out_Test"+string_group+str(i+1)+".txt", "a")
    f.write("Name: "+name[i]+"\n")
    f.write("ID: Hold_out_Test"+string_group+str(i+1)+"\n")
    f.write("Num peaks: " +num_peaks[i]+"\n")
    p = peaks[i].split(";")
    for j in range(len(p)-1):
        mass_and_intensity=p[j].split(" ")
        mass_and_intensity=[x for x in mass_and_intensity if x]
        #print(mass_and_intensity)
        peak_mass=mass_and_intensity[0]
        peak_intensity=mass_and_intensity[1]
        f.write(peak_mass+" "+peak_intensity+"\n")
    f.close()

import fileinput
import time
time.sleep(10)
def replace_in_file(file_path, search_text, new_text):
    with fileinput.input(file_path, inplace=True) as file:
        for line in file:
            new_line = line.replace(search_text, new_text)
            print(new_line, end='')

for i in range(len(smiles)):
    replace_in_file("Hold_out_Test"+string_group+str(i+1)+".txt","Name","#Name")
    replace_in_file("Hold_out_Test"+string_group+str(i+1)+".txt","ID","#ID")
    replace_in_file("Hold_out_Test"+string_group+str(i+1)+".txt","Num","#Num")
#
exit()