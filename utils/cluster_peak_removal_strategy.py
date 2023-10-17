
    mass_array = mz_array_top
    intensities = intensity_array_top
    print(len(intensities))
    print(mass_array)
    max_M = max(mass_array)
    MW_mz_list = list(range(int(max(mass_array)) + 1))  # initialize the list with 0's
    print(len(MW_mz_list))
    MW_intensity_list = [0] * len(MW_mz_list)
    for i in range(len(intensities)):
        print(int(mass_array[i]))
        MW_intensity_list[int(mass_array[i])] = int(intensities[i])


    # lists for intensities
    cluster_list = []
    temp_cluster = []

    # lists for m_by_z
    m_by_z_cluster_list = []
    m_by_z_temp_cluster = []

    flag_clust = 0
    for i in range(len(MW_intensity_list)):
        flag_clust = 0
        if MW_intensity_list[i] > 0:
            temp_cluster.append(MW_intensity_list[i])
            m_by_z_temp_cluster.append(MW_mz_list[i])
            flag_clust = 1
        if len(temp_cluster) > 0 and flag_clust == 0:
            cluster_list.append(temp_cluster)
            m_by_z_cluster_list.append(m_by_z_temp_cluster)
            temp_cluster = []
            m_by_z_temp_cluster = []

    if len(temp_cluster) > 0:
        cluster_list.append(temp_cluster)
        m_by_z_cluster_list.append(m_by_z_temp_cluster)

    new_cluster_list = []
    new_cluster_list_m_by_z = []
    # taking peak values of each cluster and deleting everything below 1 percent of that peak
    for i in range(len(cluster_list)):
        cluster_peak = max(cluster_list[i])
        threshold = cluster_peak * 0.01
        # print("printing threshold:")
        # print(threshold)
        clust_list = cluster_list[i]
        m_by_z_clust_list = m_by_z_cluster_list[i]
        # print(clust_list)
        tmp_clust = []
        tmp_mz_clust = []
        for j in range(len(clust_list)):
            if clust_list[j] > threshold:
                tmp_clust.append(clust_list[j])
                tmp_mz_clust.append(m_by_z_clust_list[j])
            # if clust_list[j] <= threshold:
            # clust_list.remove(clust_list[j])
            # m_by_z_clust_list.remove(m_by_z_clust_list[j])
        new_cluster_list.append(tmp_clust)
        new_cluster_list_m_by_z.append(tmp_mz_clust)

    #####end of commenting cluster peaks lines####
    print(new_cluster_list)
    print(new_cluster_list_m_by_z)

    combine back
    #####commenting cluster peak lines ####
    new_intensity_list = list(itertools.chain.from_iterable(new_cluster_list))
    new_m_by_z_list = list(itertools.chain.from_iterable(new_cluster_list_m_by_z))

    intensity_array_top = new_intensity_list
    mz_array_top = new_m_by_z_list
