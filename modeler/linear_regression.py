import pandas as pd
from sklearn.linear_model import LinearRegression
from data_loader.housing_price_loader import load_boston_data
import streamlit as st

df = load_boston_data()
model = LinearRegression()
model.fit(df.iloc[:, :-1], df.price)

prediction = model.predict(df.iloc[:, :-1])
df_compared = pd.DataFrame({'actual': df.price, 'prediction': prediction})
print(df_compared)
