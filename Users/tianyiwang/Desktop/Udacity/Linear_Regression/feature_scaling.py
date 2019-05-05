# TODO: Add import statements
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

# Assign the data to predictor and outcome variables
# TODO: Load the data
train_data = pd.read_csv("data.csv", header = None)
X = np.array(train_data.iloc[:,:(train_data.shape[1]-1)]).reshape(len(train_data),(train_data.shape[1]-1))
y = np.array(train_data.iloc[:,-1]).reshape(len(train_data),1)

# TODO: Create the standardization scaling object.
scaler = StandardScaler()
scaler.fit(X)

# TODO: Fit the standardization parameters and scale the data.
X_scaled = scaler.transform(X)

# TODO: Create the linear regression model with lasso regularization.
lasso_reg = Lasso().fit(X_scaled, y)

# TODO: Fit the model.


# TODO: Retrieve and print out the coefficients from the regression model.
reg_coef = lasso_reg.coef_
print(reg_coef)