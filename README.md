# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the program and import the required libraries.
2. Create the dataset with house area, number of occupants, and house price.
3. Split the dataset into training and testing sets.
4. Train the SGD Regressor model using the training data.
5. Predict the house price using test data and display the results.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by:VASHMITHA V 
RegisterNumber:  212225240180
*/
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Create dataset
data = {
    "Area": [1000, 1200, 1500, 1800, 2000, 2200, 2500, 2700, 3000, 3500],
    "Occupants": [2, 3, 4, 4, 5, 5, 6, 6, 7, 8],
    "Price": [200000, 250000, 300000, 350000, 400000, 420000, 500000, 550000, 600000, 700000]
}

df = pd.DataFrame(data)

# Step 2: Define input and output
X = df[["Area", "Occupants"]]
y = df["Price"]

# Step 3: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Train SGD Regressor model
model = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
model.fit(X_train, y_train)

# Step 5: Predict output
y_pred = model.predict(X_test)

# Step 6: Evaluate model
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Step 7: Display predictions
result = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

print("\nPrediction Results:")
print(result)
```

## Output:

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/97ad518d-7aee-420a-ae46-8a6a23d212ef" />


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
