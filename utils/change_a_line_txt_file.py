import fileinput

def replace_in_file(file_path, search_text, new_text):
    with fileinput.input(file_path, inplace=True) as file:
        for line in file:
            new_line = line.replace(search_text, new_text)
            print(new_line, end='')

import os
os.chdir('D:\\NIST20\\MSSEARCH\\pythonProject\\EI-MS_train_text_files\\Aldehydes')
for i in range(4000):
    replace_in_file("Test"+str(i+1)+".txt","Name","#Name")
    replace_in_file("Test"+str(i+1)+".txt","ID","#ID")
    replace_in_file("Test"+str(i+1)+".txt","Num","#Num")

exit()