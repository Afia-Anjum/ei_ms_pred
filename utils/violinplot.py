###Violin plots
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

tips=pd.read_csv("dist_heavy_MW_spec_entropy.csv")
print(tips)

ax=sns.violinplot(x="model",y="Spectral_entropy", data=tips, scale="count")
print(ax)
plt.show()