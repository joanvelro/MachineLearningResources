# Pandas and numpy for data manipulation
import pandas as pd
import numpy as np
np.random.seed(42)
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 16
plt.rcParams['figure.figsize'] = (9, 9)
import seaborn as sns
from IPython.core.pylabtools import figsize

# Scipy helper functions
from scipy.stats import percentileofscore
from scipy import stats

# Standard ML Models for comparison
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR

import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
warnings.filterwarnings(
    action="ignore",
    module="scipy",
    message="^.*LAPACK bug 0038.*")
# Splitting data into training/testing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

"""Metrics"""
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error


"""Distributions"""
import scipy

"""PyMC3 for Bayesian Inference"""
import pymc3 as pm
import pymc3.stats as ps

"""
code source:  https://github.com/WillKoehrsen/Data-Analysis/blob/master/bayesian_lr/Bayesian%20Linear%20Regression%20Project.ipynb
post : https://towardsdatascience.com/bayesian-linear-regression-in-python-using-machine-learning-to-predict-student-grades-part-2-b72059a8ac7e
post 2: https://towardsdatascience.com/bayesian-linear-regression-in-python-using-machine-learning-to-predict-student-grades-part-2-b72059a8ac7e
"""

# Read in class scores
df = pd.read_csv('./student-mat.csv',sep=";")

# Filter out grades that were 0
df = df[~df['G3'].isin([0, 1])]

df = df.rename(columns={'G3': 'Grade'})

df.head()

print(df.shape)

df.describe()

# Print the value counts for categorical columns
for col in df.columns:
    if df[col].dtype == 'object':
        print('\nColumn Name:', col,)
        print(df[col].value_counts())


df['Grade'].describe()

df['Grade'].value_counts()

# Bar plot of grades
if 0==1:
    plt.bar(df['Grade'].value_counts().index,
            df['Grade'].value_counts().values,
             fill = 'navy', edgecolor = 'k', width = 1)
    plt.xlabel('Grade'); plt.ylabel('Count'); plt.title('Distribution of Final Grades');
    plt.xticks(list(range(5, 20)));

# Grade distribution by address
if 0==1:
    sns.kdeplot(df[df['address'] == 'U'].Grade.values, label = 'Urban', shade = True)
    sns.kdeplot(df[df['address'] == 'R'].Grade.values, label = 'Rural', shade = True)
    plt.xlabel('Grade')
    plt.ylabel('Density')
    plt.title('Density Plot of Final Grades by Location')

# Grade distribution by Guardian
if 0==1:
    sns.kdeplot(df[df['guardian'] == 'father'].Grade.values, label='Father', shade=True)
    sns.kdeplot(df[df['guardian'] == 'mother'].Grade.values, label='Mother', shade=True)
    sns.kdeplot(df[df['guardian'] == 'other'].Grade.values, label='Other', shade=True)
    plt.xlabel('Grade')
    plt.ylabel('Density')
    plt.title('Density Plot of Final Grades by Guardian')

# Grade distribution by internet
if 0==1:
    sns.kdeplot(df[df['internet'] == 'yes'].Grade.values, label = 'Internet', shade = True)
    sns.kdeplot(df[df['internet'] == 'no'].Grade.values, label = 'No Internet', shade = True)
    plt.xlabel('Grade')
    plt.ylabel('Density')
    plt.title('Density Plot of Final Grades by Internet Access')

# Grade distribution by school
if 0==1:
    sns.kdeplot(df[df['school'] == 'GP'].Grade.values, label = 'GP', shade = True)
    sns.kdeplot(df[df['school'] == 'MS'].Grade.values, label = 'MS', shade = True)
    plt.xlabel('Grade')
    plt.ylabel('Count')
    plt.title('Distribution of Final Grades by School')

# Look at distribution of schools by address
schools = df.groupby(['school'])['address'].value_counts()
print(schools)

# Calculate percentile for grades
df['percentile'] = df['Grade'].apply(lambda x: percentileofscore(df['Grade'], x))

# Plot percentiles for grades
if 0==1:
    plt.figure(figsize = (8, 6))
    plt.plot(df['Grade'], df['percentile'], 'o')
    plt.xticks(range(0, 20, 2), range(0, 20, 2))
    plt.xlabel('Score')
    plt.ylabel('Percentile')
    plt.title('Grade Percentiles')

print('50th percentile score:', np.min(df.loc[df['percentile'] > 50, 'Grade']))
print('Minimum Score needed for 90th percentile:', np.min(df.loc[df['percentile'] > 90, 'Grade']))

# Correlations of numerical values
df.corr()['Grade'].sort_values()

# Select only categorical variables
category_df = df.select_dtypes('object')
# One hot encode the variables
dummy_df = pd.get_dummies(category_df)
# Put the grade back in the dataframe
dummy_df['Grade'] = df['Grade']
dummy_df.head()

# Correlations in one-hot encoded dataframe
dummy_df.corr()['Grade'].sort_values()


# Takes in a dataframe, finds the most correlated variables with the
# grade and returns training and testing datasets
def format_data(df):
    # Targets are final grade of student
    labels = df['Grade']

    # Drop the school and the grades from features
    df = df.drop(columns=['school', 'G1', 'G2', 'percentile'])

    # One-Hot Encoding of Categorical Variables
    df = pd.get_dummies(df)

    # Find correlations with the Grade
    most_correlated = df.corr().abs()['Grade'].sort_values(ascending=False)

    # Maintain the top 6 most correlation features with Grade
    most_correlated = most_correlated[:8]

    df = df.loc[:, most_correlated.index]
    #df = df.drop(columns='higher')

    #Split into training/testing sets with 25% split
    X_train, X_test, y_train, y_test = train_test_split(df, labels,
                                                        test_size=0.25,
                                                        random_state=42)

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = format_data(df)
X_train.head()

# Rename variables in train and teste
X_train = X_train.rename(columns={'higher_yes': 'higher_edu',
                                  'Medu': 'mother_edu',
                                  'Fedu': 'father_edu'})

X_test = X_test.rename(columns={'higher_yes': 'higher_edu',
                                  'Medu': 'mother_edu',
                                  'Fedu': 'father_edu'})


print(X_train.shape)
print(X_test.shape)

# Pairs Plot of Selected Variables
# Calculate correlation coefficient
def corrfunc(x, y, **kws):
    r, _ = stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("r = {:.2f}".format(r),
                xy=(.1, .6), xycoords=ax.transAxes,
                size=24)

if 0==1:
    cmap = sns.cubehelix_palette(light=1, dark=0.1,
                                 hue=0.5, as_cmap=True)
    sns.set_context(font_scale=2)
    # Pair grid set up
    g = sns.PairGrid(X_train)

    # Scatter plot on the upper triangle
    g.map_upper(plt.scatter, s=10, color='red')

    # Distribution on the diagonal
    g.map_diag(sns.distplot, kde=False, color='red')

    # Density Plot and Correlation coefficients on the lower triangle
    g.map_lower(sns.kdeplot, cmap=cmap)
    g.map_lower(corrfunc)


"""Create relation to the median grade column"""
X_plot = X_train.copy()
X_plot['relation_median'] = (X_plot['Grade'] >= 12)
X_plot['relation_median'] = X_plot['relation_median'].replace({True: 'above', False: 'below'})
X_plot = X_plot.drop(columns='Grade')


"""Selected Variables Distribution by Relation to Median"""
if 0==1:
    plt.figure(figsize=(12, 12))
    # Plot the distribution of each variable colored
    # by the relation to the median grade
    for i, col in enumerate(X_plot.columns[:-1]):
        plt.subplot(3, 2, i + 1)
        subset_above = X_plot[X_plot['relation_median'] == 'above']
        subset_below = X_plot[X_plot['relation_median'] == 'below']
        sns.kdeplot(subset_above[col], label='Above Median', color='green')
        sns.kdeplot(subset_below[col], label='Below Median', color='red')
        plt.legend()
        plt.title('Distribution of %s' % col)

    plt.tight_layout()

"""Establish Benchmarks"""


"""Calculate mae and rmse"""
def evaluate_predictions(predictions, true):
    mae = np.mean(abs(predictions - true))
    rmse = np.sqrt(np.mean((predictions - true) ** 2))

    return mae, rmse

"""Naive Baseline"""
median_pred = X_train['Grade'].median()
median_preds = [median_pred for _ in range(len(X_test))]
true = X_test['Grade']

"""Display the naive baseline metrics"""
mb_mae, mb_rmse = evaluate_predictions(median_preds, true)
print('Median Baseline  MAE: {:.4f}'.format(mb_mae))
print('Median Baseline RMSE: {:.4f}'.format(mb_rmse))

"""Standard Machine Learning Models"""


# Evaluate several ml models by training on training set and testing on testing set
def evaluate(X_train, X_test, y_train, y_test):
    # Names of models
    model_name_list = ['Linear Regression', 'ElasticNet Regression',
                       'Random Forest', 'Extra Trees', 'SVM',
                       'Gradient Boosted', 'Baseline']
    X_train = X_train.drop(columns='Grade')
    X_test = X_test.drop(columns='Grade')

    # Instantiate the models
    model1 = LinearRegression()
    model2 = ElasticNet(alpha=1.0, l1_ratio=0.5)
    model3 = RandomForestRegressor(n_estimators=50)
    model4 = ExtraTreesRegressor(n_estimators=50)
    model5 = SVR(kernel='rbf', degree=3, C=1.0, gamma='auto')
    model6 = GradientBoostingRegressor(n_estimators=20)

    # Dataframe for results
    results = pd.DataFrame(columns=['mae', 'rmse'], index=model_name_list)

    # Train and predict with each model
    for i, model in enumerate([model1, model2, model3, model4, model5, model6]):
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Metrics
        mae = np.mean(abs(predictions - y_test))
        rmse = np.sqrt(np.mean((predictions - y_test) ** 2))

        # Insert results into the dataframe
        model_name = model_name_list[i]
        results.loc[model_name, :] = [mae, rmse]

    # Median Value Baseline Metrics
    baseline = np.median(y_train)
    baseline_mae = np.mean(abs(baseline - y_test))
    baseline_rmse = np.sqrt(np.mean((baseline - y_test) ** 2))

    results.loc['Baseline', :] = [baseline_mae, baseline_rmse]

    return results

results = evaluate(X_train, X_test, y_train, y_test)

"""Visual Comparison of Models"""
figsize(12, 8)
plt.rcParams['font.size'] = 16
# Root mean squared error
ax =  plt.subplot(1, 2, 1)
results.sort_values('mae', ascending = True).plot.bar(y = 'mae', color = 'b', ax = ax)
plt.title('Model Mean Absolute Error')
plt.ylabel('MAE');

# Median absolute percentage error
ax = plt.subplot(1, 2, 2)
results.sort_values('rmse', ascending = True).plot.bar(y = 'rmse', color = 'r', ax = ax)
plt.title('Model Root Mean Squared Error')
plt.ylabel('RMSE');

plt.tight_layout()

print('The Gradient Boosted regressor is {:0.2f}% better than the baseline.'.format(
    (100 * abs(results.loc['Gradient Boosted', 'mae'] - results.loc['Baseline', 'mae'])) / results.loc['Baseline', 'mae']))

# Formula from Ordinary Least Squares Linear Regression

lr = LinearRegression()
lr.fit(X_train.drop(columns='Grade'), y_train)

ols_formula = 'Grade = %0.2f +' % lr.intercept_
for i, col in enumerate(X_train.columns[1:]):
    ols_formula += ' %0.2f * %s +' % (lr.coef_[i], col)

' '.join(ols_formula.split(' ')[:-1])

# Implementing Bayesian Linear Regression
# Formula for Bayesian Linear Regression (follows R formula syntax
formula = 'Grade ~ ' + ' + '.join(['%s' % variable for variable in X_train.columns[1:]])
print(formula)

# Create Model in PyMC3 and Sample from Posterior
# Context for the model
""" The model is built in a context using the with statement. In the call to GLM.from_formula we pass the formula,
the data, and the data likelihood family (this actually is optional and defaults to a normal distribution).
The function parses the formula, adds random variables for each feature (along with the standard deviation), adds 
the likelihood for the data, and initializes the parameters to a reasonable starting estimate. By default, the model 
parameters priors are modeled as a normal distribution. """

""" Once the GLM model is built, we sample from the posterior using a MCMC algorithm. If we do not specify which method, 
PyMC3 will automatically choose the best for us. In the code below, I let PyMC3 choose the sampler and specify the 
number of samples, 2000, the number of chains, 2, and the number of tuning steps, 500. """

""" In this case, PyMC3 chose the No-U-Turn Sampler and intialized the sampler with jitter+adapt_diag. """

""" The sampler runs for a few minutes and our results are stored in normal_trace. This contains all the samples 
for every one of the model parameters (except the tuning samples which are discarded). The trace is essentially 
our model because it contains all the information we need to perform inference. To get an idea of what Bayesian 
Linear Regression does, we can examine the trace using built-in functions in PyMC3. """
with pm.Model() as normal_model:
    # The prior for the model parameters will be a normal distribution
    family = pm.glm.families.Normal()

    # Creating the model requires a formula and data (and optionally a family)
    pm.GLM.from_formula(formula, data=X_train, family=family)

    # Perform Markov Chain Monte Carlo sampling
    normal_trace = pm.sample(draws=2000, chains=2, tune=500, progressbar=1)



# Examine Bayesian Linear Regression Results
# Traceplot of All Samples
# Shows the trace with a vertical line at the mean of the trace
""" A traceplot shows the posterior distribution for the model parameters (on the left) and the progression of the samples 
drawn in the trace for the variable (on the right). The two colors represent the two difference chains sampled. """
def plot_trace(trace):
    # Traceplot with vertical lines at the mean value
    ax = pm.traceplot(trace, figsize=(14, len(trace.varnames) * 1.8),
                      lines={k: v['mean'] for k, v in ps.df_summary(trace).iterrows()})

    plt.rcParams['font.size'] = 16

    # Labels with the median value
    for i, mn in enumerate(ps.df_summary(trace)['mean']):
        ax[i, 0].annotate('{:0.2f}'.format(mn), xy=(mn, 0), xycoords='data', size=8,
                          xytext=(-18, 18), textcoords='offset points', rotation=90,
                          va='bottom', fontsize='large', color='red')

if 1==1:
    plot_trace(normal_trace)

"""Another way to look at the posterior distributions is as histograms"""
"""The left side of the traceplot is the marginal posterior: the values for the variable are on the x-axis with the 
probability for the variable (as determined by sampling) on the y-axis. The different colored lines indicate that we
performed two chains of Markov Chain Monte Carlo. From the left side we can see that there is a range of values for 
each weight. The right side shows the different sample values drawn as the sampling process runs.
Another method built into PyMC3 for examinig trace results is the forestplot which shows the distribution of each 
sampled parameter. This allows us to see the uncertainty in each sample. The forestplot is easily constructed from the
trace using pm.forestplot."""
if 1==1:
    pm.traceplot(normal_trace)