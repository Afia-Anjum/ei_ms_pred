import pandas as pd
import time
import itertools
import matplotlib.pyplot as plt

# 3D scatter plots:

train = pd.read_csv('test_results_der_stdpolar.csv', sep=',', error_bad_lines=False, index_col=False, dtype='unicode')
ri = train['RI']
ri_array = ri.to_numpy()
logp = train['LogP']
logp_array = logp.to_numpy()
em = train['ExactMass']
mass_array = em.to_numpy()
rb = train['NumberofRotatableBonds']
rb_array = rb.to_numpy()

ri_array = ri_array.astype('float')
logp_array = logp_array.astype('float')
mass_array = mass_array.astype('float')
rb_array = rb_array.astype('float')

ax = plt.axes(projection='3d')

# Data for a three-dimensional line
# zline = np.linspace(0, 15, 100)
# xline = np.sin(zline)
# yline = np.cos(zline)

# ax.plot3D(xline, yline, zline, 'gray')

# Data for three-dimensional scattered points

ax.scatter3D(mass_array, rb_array, ri_array, c=ri_array, cmap='Greens');
ax.set_xlabel('ExactMass')
ax.set_ylabel('RotatableBonds')
ax.set_zlabel('RI')
plt.show()
