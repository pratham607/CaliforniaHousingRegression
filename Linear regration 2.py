import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
df=pd.read_csv("Cleaned housing data")
df.copy()
le=LabelEncoder()
df["ocean_proximity_encoded"]=le.fit_transform(df["ocean_proximity"])
print(df[["ocean_proximity","ocean_proximity_encoded"]])
X=df[["longitude","latitude","housing_median_age","total_rooms","total_bedrooms","population","households","median_income","ocean_proximity_encoded"]]
y=df["median_house_value"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(X_train,y_train)
new_pred=model.predict(X_test)
new_pred[5]
print(new_pred)
plt.scatter(X_train["median_income"],y_train,color="blue", alpha=0.5, label="Training data")
plt.scatter(X_test["median_income"],new_pred,color="red", alpha=0.5, label="Predictions")
plt.title("Housing price prediction model")
plt.xlabel("Input conditions ")
plt.ylabel("Prediction")
plt.savefig("Output of linear regrationn model")
plt.show()
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, new_pred)
r2 = r2_score(y_test, new_pred)
print(f"MSE: {mse}, R2 Score: {r2}")
"""plt.scatter(X_train["median_income"], y_train, color="blue", alpha=0.5, label="Training data")
plt.scatter(X_test["median_income"], new_pred, color="red", alpha=0.5, label="Predictions")
plt.title("Housing price prediction model")
plt.xlabel("Median Income")
plt.ylabel("House Value")
plt.legend()
plt.show()"""
