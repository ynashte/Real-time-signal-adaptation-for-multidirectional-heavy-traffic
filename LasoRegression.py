import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

data = pd.read_csv("D://Mini Project Sem 5//traffic_signal_data.csv")

X = data[['noOfVehicles']].values
y = data['Time'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lasso1 = Lasso(alpha=0.5)
lasso1.fit(X_train, y_train)

y_pred = lasso1.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.2f}")


plt.scatter(X_test, y_test, color='blue', label='Actual Data')
plt.plot(X_test, y_pred, color='red', label='Predicted Data')
plt.xlabel("Number of Vehicles")
plt.ylabel("Traffic Light Timing")
plt.title("Lasso Regression for Traffic Light Timing Prediction")
plt.legend()
plt.show()
predicted_time = lasso1.predict([[15]])
print("Predicted Time for 150 vehicles is:", predicted_time[0])
