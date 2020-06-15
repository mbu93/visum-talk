from joblib import load


class MnistModel:
    def __init__(self):
        print("Invoking model")
        path = "/pv/clf.pickle"
        self.model = load(path)

    def predict(self, x, shape):
        print(x)
        print(shape)
        return self.model.predict(x.reshape(-1, 8 * 8))
