import sys
import os

sys.path.append(os.getcwd())

import pandas as pd
import matplotlib.pyplot as plt

from utils.data_loader import load_data
from utils.data_preprocessor import preprocess_data
from utils.model_evaluator import evaluate_model
from models.model_factory import get_model

data = load_data("data/raw/traffic_accidents.csv")

X_train, X_test, y_train, y_test = preprocess_data(data)

models = ["rf", "svm", "knn", "nb"]
accuracies = []

for m in models:
    model = get_model(m)

    model.train(X_train, y_train)

    pred = model.predict(X_test)

    acc, pre, rec, f1 = evaluate_model(y_test, pred)

    accuracies.append(acc)

plt.bar(models, accuracies)

plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")

plt.savefig("model_comparison.png")

print("Model comparison graph saved!")
