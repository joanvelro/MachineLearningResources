import pandas as pd
import numpy as np
import seaborn as sns
sns.set(style="ticks")
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


import seaborn as sns


"""   []   """

"""read in the dataset"""
df = pd.read_csv('data/diabetes_data.csv')

"""take a look at the data"""
print(df.head())

"""check dataset size"""
print(df.shape)
n,m = df.shape



"""normalise data"""
scaler = MinMaxScaler()
scaler.fit(df.values)
data_norm = scaler.transform(df.values)
df_norm = pd.DataFrame(columns = df.columns.values, data=data_norm)
df0 = df_norm.iloc[:,0:m-2]

"""boxplot"""
if 1==1:
    fig = plt.figure(num=None, figsize=(8, 6), dpi=60, facecolor='w', edgecolor='k')
    a = df0.boxplot()
    plt.savefig("boxplot.pdf")

"""split data into inputs and targets"""
X = df0
y = df['diabetes']

"""split data into train and test sets"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)


"""create new a knn model"""
knn = KNeighborsClassifier()
"""create a dictionary of all values we want to test for n_neighbors"""
params_knn = {'n_neighbors': np.arange(1, 25)}
"""use gridsearch to test all values for n_neighbors"""
knn_gs = GridSearchCV(knn, params_knn, cv=5)
"""fit model to training data"""
knn_gs.fit(X_train, y_train)
"""save best model"""
knn_best = knn_gs.best_estimator_
"""check best n_neigbors value"""
print(knn_gs.best_params_)


"""Create the default pairplot"""
if 1==1:
    sns_plot = sns.pairplot(df0, height=1.5)
    sns_plot.savefig("scatterplot.pdf")
    plt.close()



if 0==1:
    """Function to calculate correlation coefficient between two arrays"""
    def corr(x, y, **kwargs):
        # Calculate the value
        coef = np.corrcoef(x, y)[0][1]
        # Make the label
        label = r'$\rho$ = ' + str(round(coef, 2))

        # Add the label to the plot
        ax = plt.gca()
        ax.annotate(label, xy=(0.2, 0.95), size=20, xycoords=ax.transAxes)
    """Create an instance of the PairGrid class."""
    grid = sns.PairGrid(data=df0,
                        vars=['pregnancies', 'glucose',
                              'diastolic'],
                        height=4)
    """Map a scatter plot to the upper triangle"""
    grid = grid.map_upper(plt.scatter, color='darkred')
    grid = grid.map_upper(corr)
    """Map a histogram to the diagonal"""
    grid = grid.map_diag(plt.hist, bins=10, color='darkred',
                         edgecolor='k')
    """Map a density plot to the lower triangle"""
    grid = grid.map_lower(sns.kdeplot, cmap='Reds')