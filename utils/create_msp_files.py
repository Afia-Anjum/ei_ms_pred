import os
os.chdir('D:\\NIST20\\MSSEARCH\\pythonProject\\EI-MS_train_text_files\\Aldehydes')


with open('mergedfile_Aldehydes_400_to_500.txt', 'w') as newfile:
  for i in range(400,500):
      with open("Test"+str(i+1)+".txt") as infile:
        contents = infile.read()
        newfile.write(contents)
        newfile.write("\n")
exit()