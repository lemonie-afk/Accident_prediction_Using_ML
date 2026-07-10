import pandas as pd
from sklearn.model_selection import train_test_split
import os

def preprocess_data(data):

    data = data.dropna()

    data = pd.get_dummies(data)

    os.makedirs("data/processed", exist_ok=True)

    data.to_csv("data/processed/cleaned_accidents.csv", index=False)

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test
