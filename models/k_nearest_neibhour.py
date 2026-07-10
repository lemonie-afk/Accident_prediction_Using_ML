from sklearn.neighbors import KNeighborsClassifier

class KNNModel:

    def train(self, X_train, y_train):
        self.model = KNeighborsClassifier()
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
