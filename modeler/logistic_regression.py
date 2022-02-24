import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from data_loader.housing_price_loader import load_cancer_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utils.chart import plot_confusion_matrix
warnings.filterwarnings('ignore')

df = load_cancer_data()
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.iloc[:, -1], test_size=0.15, shuffle=True,
                                                    random_state=4)

model = LogisticRegression(penalty='l1', C=50, solver='saga')
model.fit(X_train, y_train)

# prediction = model.predict(X_test)  # p >= 0.5 --> 1.
prediction = model.predict_proba(X_test)[:, 1]
classes = np.where(prediction >= 0.45, 1, 0)

df_compared = pd.DataFrame({'actual': y_test, 'prediction': classes})

plot_confusion_matrix(df_compared.actual, df_compared.prediction)
print(classification_report(df_compared.actual, df_compared.prediction))
# df.cancer.map({'cancer': 1, 'non_cancer': 0})
