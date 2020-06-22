import numpy as np
from pyHSICLasso import HSICLasso as HLasso


class HSICLasso:
    def __init__(self, k=10):
        self.model = HLasso()
        self.k = k
        self.modelname = "HSICLasso_{}".format(k)
    
    def fit(self, X, y):
        self.model.input(X, y)
        self.model.classification(self.k)

        self.index = np.array(self.model.get_index())

        return self
        
    def transform(self, X):
        return X[:, self.index]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
