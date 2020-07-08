"""
HOUSE PRICE PREDICTION USING TENSORFLOW:

"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('DATA/kc_house_data.csv')
print(df.head())

print(df.columns)

print(df.info())

print(df.describe().T)

# To Check For Null Values


print(df.isnull().sum())

# Exploratory Data Analysis :

sns.set_style('darkgrid')
sns.distplot(df['price'])
plt.show()

sns.countplot(df['bedrooms'])
plt.show()

sns.scatterplot(x='price', y='sqft_living', data=df)
plt.show()

sns.boxplot(x='bedrooms', y='price', data=df)
plt.show()

sns.scatterplot(y='lat', x='long', hue='price', data=df)
plt.show()

print(df.corr()['price'].sort_values(ascending=False))

sns.heatmap(df.corr())
plt.show()

print(df.sort_values('price', ascending=False).head())

sns.boxplot(y='price', x='waterfront', data=df)
plt.show()

""" 
Feature Engineering From Date :
"""

print(df['date'])

df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].apply(lambda date: date.month)
df['year'] = df['date'].apply(lambda date: date.year)

sns.boxplot(x='year', y='price', data=df)
plt.show()

sns.boxplot(x='month', y='price', data=df)
plt.show()

df.groupby('month').mean()['price'].plot()
plt.show()

df.groupby('year').mean()['price'].plot()
plt.show()

df['yr_renovated'].value_counts(ascending=False)

""" 
DROPPING SOME NOT USEFUL FEATURES :
"""

df = df.drop(['id', 'date', 'zipcode'], axis=1)

print(df.columns)

""" 
Defining Our Features And Labels : 
"""

X = df.drop('price', axis=1).values
y = df['price'].values

""" 
TRAIN TEST SPLIT & SCALING OUR DATA :

"""

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Shape of Features of X_train For Neural Network : ", X_train.shape)

"""
CREATING A NEURAL NETWORK MODEL : 
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam

model = Sequential()

model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

""" 
TRAINING OUR MODEL :
"""

model.fit(x=X_train, y=y_train,
          validation_data=(X_test, y_test), batch_size=128,
          epochs=400)

# Checking The Plot for our losses :

losses = pd.DataFrame(model.history.history)
losses.plot()
plt.show()

predictions = model.predict(X_test)

print("y_test shape:", y_test.shape)

print("predictions shape :", predictions.shape)

predictions = predictions.reshape(6480, )

df_comb = pd.DataFrame({'Actual Values': y_test, "Predicted Values": predictions})
print(df_comb.head(20))

plt.scatter(y_test,predictions)
plt.show()
""" 
Evaluation :
"""

from sklearn.metrics import mean_absolute_error, mean_squared_error,explained_variance_score

print("Mean Absolute Error :", mean_absolute_error(y_test, predictions))
print('\n')
print("Mean Squared Error :", mean_squared_error(y_test, predictions))
print('\n')
print("Variance Score :", explained_variance_score(y_test, predictions))

Errors=y_test-predictions
sns.distplot(Errors)
plt.show()

""" 
Predicting A New House Price With New Features :

Let Us Suppose We Have A New House With The Following Features :


bedrooms  bathrooms  sqft_living  sqft_lot  floors  waterfront  view  condition  grade  sqft_above  sqft_basement  


4	       2.00	       1200	       6000	     1.0	  0	          0	       5	   8	   1220	        900	        


yr_built  yr_renovated  lat        long    sqft_living15  sqft_lot15  month  year

1980	     0	      47.5102	-122.258	    1400	      5000	      10	2014





Let us try to predict this house's price :

"""
print('\n')
new_house_feat = [[4, 2.00, 1200, 6000, 2.0, 0, 0, 5, 8, 1220, 900, 1980, 0, 47.5102, -122.258, 1400, 5000, 10, 2014]]

new_house_feat=scaler.transform(new_house_feat)

new_house_feat_pred=model.predict(new_house_feat)

print('Predicted House Price Is : ', new_house_feat_pred)
