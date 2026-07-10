from sklearn.ensemble import RandomForestClassifier


class RandomForestModel:

    def train(self, X_train, y_train):
        self.model = RandomForestClassifier()
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
