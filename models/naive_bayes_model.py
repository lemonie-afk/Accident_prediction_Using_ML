from sklearn.naive_bayes import GaussianNB
class NaiveBayesModel:

    def train(self, X_train, y_train):
        self.model = GaussianNB()
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
