import warnings
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression, LogisticRegression
from data_loader.housing_price_loader import load_boston_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


warnings.filterwarnings('ignore')
df = load_boston_data()
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.iloc[:, -1], test_size=0.15, shuffle=True,
                                                    random_state=4)

model = LinearRegression()
model.fit(X_train, y_train)

prediction = model.predict(X_test)
np.savetxt('../Live_Lecture/prediction.csv', prediction)

df_compared = pd.DataFrame({'actual': y_test, 'prediction': prediction})
print(f"Mean squared error, {mean_squared_error(df_compared.actual, df_compared.prediction)}")
print(f"Mean absolute error, {mean_absolute_error(df_compared.actual, df_compared.prediction)}")
print(f"Coefficient of determination, {r2_score(df_compared.actual, df_compared.prediction)}")

# with open('../saved_model/housing_price_linear.pickle', 'wb') as f:
#     pickle.dump(model, f)

with open('../saved_model/housing_price_linear.pickle', 'rb') as f:
    model_2 = pickle.load(f)

print('DONE')
