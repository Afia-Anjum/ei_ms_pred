M_intensity = []
for i in range(len(final_MW)):
    tmp = final_peaks[i].split(";")
    tmp.pop()
    print(tmp)
    exit()
    flag = 0
    for j in range(len(tmp)):
        new_tmp = tmp[j].split()
        print(new_tmp[0])
        print(final_MW[i])
        exit()
        if int(new_tmp[0]) == final_MW[i]:
            M_intensity.append(float(new_tmp[1]))
            flag = 1
            break
    if flag == 0:
        M_intensity.append(float(new_tmp[1]))
