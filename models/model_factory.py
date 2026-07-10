from models.random_forest_model import RandomForestModel
from models.support_vector_machine import SVMModel
from models.k_nearest_neighbour import KNNModel
from models.naive_bayes_model import NaiveBayesModel


def get_model(name):

    if name == "rf":
        return RandomForestModel()

    elif name == "svm":
        return SVMModel()

    elif name == "knn":
        return KNNModel()

    elif name == "nb":
        return NaiveBayesModel()

    else:
        raise ValueError("Invalid Model")
