import numpy as np
import pandas as pd

import pymrmr

from ..tools import silence


class mRMR:
    def __init__(self, k=10):
        self.k = k
        self.modelname = "mRMR_{}".format(k)
    
    def fit(self, X, y):
        n, d = X.shape
        
        # Creating a dataFrame
        vectors = np.concatenate([y[:, None], X], axis=1)
        columns = ["label"] + [str(x) for x in range(d)]
        df = pd.DataFrame(vectors, columns=columns)
        
        with silence():
            output = pymrmr.mRMR(df, 'MIQ', self.k)
        
        self.index = np.array([int(x) for x in output])
        
        return self
        
    def transform(self, X):
        return X[:, self.index]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
