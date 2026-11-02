# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the required libraries NumPy and Scikit-learn.
2. Create the input dataset with house features and output values (house price and number of occupants).
3. Create and train the SGD Regressor model using the training data.
4. Predict the house price and number of occupants using the trained model.
5. Display the predicted results.
6. Verify the model performance using the given dataset.

## Program:
```

import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

X = np.array([
    [1000, 2, 5],
    [1500, 3, 4],
    [2000, 3, 3],
    [2500, 4, 2],
    [3000, 4, 1]
])

Y = np.array([
    [20, 2],
    [30, 3],
    [40, 4],
    [50, 5],
    [60, 6]
])

model = make_pipeline(
    StandardScaler(),
    MultiOutputRegressor(SGDRegressor(max_iter=5000, tol=1e-4, random_state=42))
)

model.fit(X, Y)

new_house = np.array([[2200, 3, 2]])

prediction = model.predict(new_house)

print("Predicted House Price:", round(prediction[0][0], 2))
print("Predicted Number of Occupants:", round(prediction[0][1], 2))

```

## Output:
<img width="600" height="117" alt="image" src="https://github.com/user-attachments/assets/f14403f9-f3ef-4e72-b670-84022b6c6c15" />


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
