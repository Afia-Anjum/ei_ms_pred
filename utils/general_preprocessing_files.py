from os import path
import os.path
###DEAL WITH EI-2.0 Files###
os.chdir('D:\\NIST20\\MSSEARCH\\pythonProject\\experiment\\output_EI_nist23_test_Jan28_hold_out\\')
import math
input_start=3535
#input_start=2501
arr_input=[]
#for k in range(3535,3680):
#for k in range(1,2009):
for k in range(1, 21):
    if path.exists("Hold_out_Test_nist23_"+str(k)+".log"):
        arr_input.append(k)
#for k in range(2501,2601):
#for k in range(2501, 2502):
#    arr_input.append(k)
#print(arr_input)
for k in range(len(arr_input)):
    with open("Hold_out_Test_nist23_"+str(arr_input[k])+".log") as f:
        content_list = f.readlines()
    content_list = [(x.strip()) for x in content_list]
    content_list.pop(0)
    #print(content_list)
    v=content_list.index("")
    content_list = content_list[:v]
    #print(content_list)

    mass=[]
    intensity=[]
    for i in range(len(content_list)):
        p_index=content_list[i].split()
        mass.append(int(float(p_index[0])))
        intensity.append(p_index[1])

    #print(mass)
    #print(intensity)
    correct_raw_m_by_z=[]
    correct_raw_intensity=[]

    for l in range(len(mass)):
        if mass[l] not in correct_raw_m_by_z:
            correct_raw_m_by_z.append(mass[l])
            correct_raw_intensity.append(float(intensity[l]))
        else:
            m=len(correct_raw_intensity)
            correct_raw_intensity[m-1]=str(float(correct_raw_intensity[m-1])+float(intensity[l]))

    #print(correct_raw_m_by_z)
    #print(correct_raw_intensity)

    with open("Hold_out_Test_Ketone_"+str(arr_input[k])+".txt", "a") as file_object:
        # Append 'hello' at the end of file
        for i in range(len(correct_raw_m_by_z)):
            file_object.write(str(correct_raw_m_by_z[i])+" "+str(correct_raw_intensity[i])+"\n")


###DEAL WITH new EIMS Files###

import math
input_start=3535
#input_start=2501
arr_input=[]
#for k in range(3501,3601):
# for k in range(1,68):
# #for k in range(2501, 2601):
#     arr_input.append(k)
#for k in range(3535,3680):
for k in range(1,102):
    if path.exists("Predicted_Test"+str(k)+".log"):
        arr_input.append(k)
#print(arr_input)
for k in range(len(arr_input)):
    with open("Predicted_Test"+str(arr_input[k])+".log") as f:
        content_list = f.readlines()
    content_list = [(x.strip()) for x in content_list]
    len_loop=8
    for loop in range(len_loop):
        content_list.pop(0)
    #content_list.pop(0)
    #print(content_list)
    v=content_list.index("")
    content_list = content_list[:v]
    #print(content_list)

    mass=[]
    intensity=[]
    for i in range(len(content_list)):
        p_index=content_list[i].split()
        mass.append(p_index[0])
        intensity.append(p_index[1])

    #print(mass)
    #print(intensity)

    with open("Predicted_Test"+str(arr_input[k])+".txt", "a") as file_object:
        # Append 'hello' at the end of file
        for i in range(len(mass)):
            file_object.write(str(mass[i])+" "+str(intensity[i])+"\n")


###DEAL WITH NIST Original ground truth files
os.chdir('D:\\NIST20\\MSSEARCH\\pythonProject\\experiment\\')
import math
input_start=3535
#input_start=2501
#arr_input=[]
#for k in range(3501,3601):
# for k in range(1,68):
# #for k in range(2501, 2601):
#     arr_input.append(k)
#for k in range(3535,3680):
# for k in range(1,355):
#     if path.exists("Hold_out_Test"+str(k)+"_orig.txt"):
#         arr_input.append(k)
#print(arr_input)
for k in range(len(arr_input)):
    with open("Hold_out_Test_Ketone_"+str(arr_input[k])+"_orig.txt",'r') as f:
        content_list = f.readlines()
    with open("Hold_out_Test_Ketone_"+str(arr_input[k])+"_orig.txt",'w') as f:
        for number, line in enumerate(content_list):
            if number not in [0, 1, 2]:
                f.write(line)


#########ONLY for EI-2 ########
arr_input=[42,51,64,65,68,69,75,78,81,90,91,93,97,98,99]
for k in range(len(arr_input)):
    with open("Mol"+str(arr_input[k])+".txt") as f:
        content_list_raw = f.readlines()
    content_list_raw = [(x.strip()) for x in content_list_raw]
    content_list_raw.remove(content_list_raw[0])
    target_index=content_list_raw.index('')
    content_list_raw=content_list_raw[:target_index]

    raw_m_by_z=[]
    raw_intensity=[]

    for l in range(len(content_list_raw)):
        p=content_list_raw[l].split()
        print(p)
        raw_m_by_z.append(int(float(p[0])))
        raw_intensity.append(p[1])

    print(raw_m_by_z)
    correct_raw_m_by_z=[]
    correct_raw_intensity=[]

    for l in range(len(raw_m_by_z)):
        if raw_m_by_z[l] not in correct_raw_m_by_z:
            correct_raw_m_by_z.append(raw_m_by_z[l])
            correct_raw_intensity.append(raw_intensity[l])
        else:
            m=len(correct_raw_intensity)
            correct_raw_intensity[m-1]=str(float(correct_raw_intensity[m-1])+float(raw_intensity[l]))

    print(correct_raw_m_by_z)
    print(correct_raw_intensity)

    # Open a file with access mode 'a'
    with open("Test"+str(arr_input[k])+".txt", "a") as file_object:
        # Append 'hello' at the end of file
        for i in range(len(correct_raw_m_by_z)):
            file_object.write(str(correct_raw_m_by_z[i])+" "+str(correct_raw_intensity[i])+"\n")
