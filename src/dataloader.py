import numpy as np
import random
from scipy.io import loadmat

from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split


def split_data(X, y, seed):
    idx = random.sample(range(0, X.shape[0]), round(X.shape[0]*0.5))
    X_train = X[idx, :]
    y_train = y[idx, :]
    X_test = np.delete(X, idx, 0)
    y_test = np.delete(y, idx, 0)
    return X_train, X_test, y_train, y_test


class DataLoader:
    def __init__(self, name, normalization=False, seed=1):
        mat = loadmat(name)        
        split = split_data(mat["X"], mat["Y"], seed)
        X_train_original, X_test_original, y_train, y_test = split

        self.y_train = np.ravel(y_train)
        self.y_test = np.ravel(y_test)
        
        if normalization:
            scaler = StandardScaler()
            scaler.fit(X_train_original)
            self.X_train = scaler.transform(X_train_original)
            self.X_test = scaler.transform(X_test_original)
        else:
            self.X_train = X_train_original
            self.X_test = X_test_original


# All datasets available at http://featureselection.asu.edu/datasets.php

class ALLAMLLoader(DataLoader):
    def __init__(self, **kargs):
        name = "data/ALLAML.mat"
        super().__init__(name, **kargs)


class CLL_SUB_111LLoader(DataLoader):
    def __init__(self, **kargs):
        name = "data/CLL_SUB_111.mat"
        super().__init__(name, **kargs)


class GLI_85Loader(DataLoader):
    def __init__(self, **kargs):
        name = "data/GLI_85.mat"
        super().__init__(name, **kargs)


class GLIOMALoader(DataLoader):
    def __init__(self, **kargs):
        name = "data/GLIOMA.mat"
        super().__init__(name, **kargs)


class Prostate_GELoader(DataLoader):
    def __init__(self, **kargs):
        name = "data/Prostate_GE.mat"
        super().__init__(name, **kargs)


class SMK_CAN_187Loader(DataLoader):
    def __init__(self, **kargs):
        name = "data/SMK_CAN_187.mat"
        super().__init__(name, **kargs)


DataLoaders = {"ALLAML": ALLAMLLoader,
               "CLL_SUB": CLL_SUB_111LLoader,
               "GLI_85": GLI_85Loader,
               "GLIOMA": GLIOMALoader,
               "Prostate_GE": Prostate_GELoader,
               "SMK_CAN": SMK_CAN_187Loader}
