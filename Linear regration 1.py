import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
df=pd.read_csv("Cleaned housing data")
df.copy()
le=LabelEncoder()
df["ocean_proximity_encoded"]=le.fit_transform(df["ocean_proximity"])
print(df[["ocean_proximity","ocean_proximity_encoded"]])
X=df[["longitude","latitude","housing_median_age","total_rooms","total_bedrooms","population","households","median_income","ocean_proximity_encoded"]]
y=df["median_house_value"]
model=LinearRegression()
model.fit(X,y)
new_pred=model.predict(X)
new_pred[5]
print(new_pred)
plt.scatter(df["median_income"],y )
plt.scatter(df["median_income"],new_pred)
plt.title("Housing price prediction model")
plt.xlabel("Input conditions ")
plt.ylabel("Prediction")
plt.show()