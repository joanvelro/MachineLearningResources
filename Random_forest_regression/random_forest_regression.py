"""https://pythondata.com/forecasting-with-random-forests/"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, explained_variance_score
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score, train_test_split

"""lets set the figure size and color scheme for plots"""
"""personal preference and not needed)."""
plt.rcParams['figure.figsize'] = (20, 10)
plt.style.use('ggplot')

"""read data"""
df = pd.read_csv('./data/Housing.csv')

"""plot data"""
df = df[['price', 'lotsize']]
df.plot(subplots=True)

"""select X and y data"""
X = (df['lotsize'])
y = (df['price'])

if 1 == 1:
    X_train = X[X.index < 400]
    y_train = y[y.index < 400]

    X_test = X[X.index >= 400]
    y_test = y[y.index >= 400]

"""build model """
rfm = RandomForestRegressor(n_estimators=100,
                            max_features=1,
                            oob_score=True)
"""fit model"""
rf = rfm.fit(X_train[:, None], y_train)

"""make predictions"""
y_hat_test = pd.Series(rf.predict(X_test[:, None]))
y_hat_train = pd.Series(rf.predict(X_train[:, None]))
y_hat = pd.concat((y_hat_train, y_hat_test), axis=0).reset_index(drop=True)
df['predicted_price'] = y_hat
df['Error'] = 100 * (df['predicted_price'] - df['price']) / df['price']

"""plot predictions"""
df[['price', 'predicted_price']].plot()

"""plot errors"""
plt.figure()
df['Error'].plot()

"""Obtain regression metris"""
print('R2:', r2_score(df['price'], df['predicted_price']))
print('MSE:', mean_squared_error(df['price'], df['predicted_price']))
print('MAE:', mean_absolute_error(df['price'], df['predicted_price']))
print('EV:', explained_variance_score(df['price'], df['predicted_price']))




