# TODO: Add import statements
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso

# Assign the data to predictor and outcome variables
# TODO: Load the data
train_data = pd.read_csv("data.csv", header = None)
X = np.array(train_data.iloc[:,:(train_data.shape[1]-1)]).reshape(len(train_data),(train_data.shape[1]-1))
y = np.array(train_data.iloc[:,-1]).reshape(len(train_data),1)

# TODO: Create the linear regression model with lasso regularization.
l = Lasso()
lasso_reg = l.fit(X,y)

# TODO: Fit the model.


# TODO: Retrieve and print out the coefficients from the regression model.
reg_coef = l.coef_
print(reg_coef)