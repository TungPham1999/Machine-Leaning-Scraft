import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


df = pd.read_csv('FuelConsumption.csv', encoding = "ISO-8859-1")

# take a look at the dataset
df.head()

# summarize the data
df.describe()

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)


viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
viz.hist()

# lets plot each of these features vs the Emission, to see how linear is their relation:
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")

#with engine size
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='red')
plt.xlabel("Engine size")
plt.ylabel("Emission")

#with cylinder 
plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS,  color='yellow')
plt.xlabel("Cylinders")
plt.ylabel("Emission")

# random for test 80% training 20% test
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")

from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']]) # x bar hay là theta 1 hay coefficient
train_y = np.asanyarray(train[['CO2EMISSIONS']]) # the ra 0 hay intercept
regr.fit (train_x, train_y)
# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)

# can plot the fit line over the data:
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")

# part test 
from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_hat = regr.predict(test_x) # predict

#train hat
train_y_hat = regr.predict(train_x)

print("Residual predict sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))
print("Residual accuary sum of squares (MSE): %.2f" % np.mean((train_y_hat - train_y) ** 2))
