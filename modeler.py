from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
from config import SEED


class Modeller:
    def __init__(self, x_train, y_train, model_type):
        self.x_train = x_train
        self.y_train = y_train
        self.model_type = model_type


    def build_model(self,):
        model = self.build_lgbm(self.x_train, self.y_train, self.model_type)

        return model


    def build_lgbm(self, x_train, y_train, model_type):
        if model_type =='regressor':
            model = LGBMRegressor(random_state=SEED)
        else:
            model = LGBMClassifier(random_state=SEED)

        model.fit(x_train, np.ravel(y_train))

        return model
        