import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score


# read dataset

df = pd.read_csv("FuelConsumption.csv")

# take a look at the dataset
df.head()

# select some features that we want to use for regression

cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY',
          'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
cdf.head(9)

# random for test 80% training 20% test
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]


# ENGINESIZE

# add model
regr = linear_model.LinearRegression()
# x bar hay là theta 1 hay coefficient
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])  # the ra 0 hay intercept
regr.fit(train_x, train_y)

# can plot the fit line over the data:
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")


# plot Emission values with respect to Engine size
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color="blue")
plt.xlabel("Engine size")
plt.ylabel("Emission")

# residual engine size


test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_hat = regr.predict(test_x)


# CYLINDERS


# train
train_x1 = np.asanyarray(train[['CYLINDERS']])
train_y1 = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x1, train_y1)


# test

test_x1 = np.asanyarray(test[['CYLINDERS']])
test_y1 = np.asanyarray(test[['CO2EMISSIONS']])
test_y1_hat = regr.predict(test_x1)


# FUELCONSUMPTION_CITY

# train
train_x2 = np.asanyarray(train[['FUELCONSUMPTION_CITY']])
train_y2 = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x2, train_y2)

# test
test_x2 = np.asanyarray(test[['FUELCONSUMPTION_CITY']])
test_y2 = np.asanyarray(test[['CO2EMISSIONS']])
test_y2_hat = regr.predict(test_x2)


# FUELCONSUMPTION_HWY
# train
train_x3 = np.asanyarray(train[['FUELCONSUMPTION_HWY']])
train_y3 = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x3, train_y3)

# test
test_x3 = np.asanyarray(test[['FUELCONSUMPTION_HWY']])
test_y3 = np.asanyarray(test[['CO2EMISSIONS']])
test_y3_hat = regr.predict(test_x3)


# Multiple
regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_HWY']])
y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(x, y)

y_hat = regr.predict(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_HWY']])
x = np.asanyarray(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_HWY']])
y = np.asanyarray(test[['CO2EMISSIONS']])


# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(x, y))

model_array = [np.mean((test_y_hat - test_y) ** 2), np.mean((test_y1_hat - test_y1) ** 2),
               np.mean((test_y2_hat - test_y2) ** 2), np.mean((test_y3_hat - test_y3) ** 2)]

model_best = model_array[0]
for i in range(len(model_array)):
      if model_array[i] < model_best:
            model_best = model_array[i]
      else:
            i = i +1
print(model_best)
