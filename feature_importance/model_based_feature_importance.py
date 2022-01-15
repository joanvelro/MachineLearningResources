import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
# []{}
names_columns = ['vendor_name','model_name','MYCT','MMIN','MMAX','CACH','CHMIN','CHMAX','PRP','ERP']
df = pd.read_csv('machine.data.csv',header=0,names=names_columns)

R = df.corr()

fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(R, vmax=1.0, center=0, fmt='.2f',
                square=True, linewidths=.5, annot=True,cmap = cm.get_cmap('Greys') , cbar_kws={"shrink": .70})
plt.show();