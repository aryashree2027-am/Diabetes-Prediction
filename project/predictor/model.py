import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def train_model():
    data = pd.read_csv("diabetes.csv")

    data = data.drop(['Pregnancies',"DiabetesPedigreeFunction"], axis=1)

    cols = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI','Age']
    data[cols] = data[cols].replace(0, np.nan)

    for col in cols:
        data[col].fillna(data[col].mean(), inplace=True)

    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)

    return model