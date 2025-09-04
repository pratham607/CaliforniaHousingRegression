import pandas as pd
df=pd.read_csv("archive (3).zip")
df["total_bedrooms"]=df["total_bedrooms"].fillna(df["total_bedrooms"].mean())
#print(df["total_bedrooms"].tail())
print(df.info())
df.to_csv("Cleaned housing data")