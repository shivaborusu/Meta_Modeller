from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
from config import SEED


class Modeller:
    def __init__(self, x_train, y_train, x_test, y_test, model_type):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.model_type = model_type


    def build_model(self):
        #build model 1 - LGBM
        model_1 = self.build_model_1(self.x_train, self.y_train, self.model_type)

        #build model 1
        model_2 = self.build_model_2(self.x_train, self.y_train, self.model_type)

        #build model 1
        model_3 = self.build_model_3(self.x_train, self.y_train, self.model_type)

        #build model 1
        model_4 = self.build_model_4(self.x_train, self.y_train, self.model_type)

        best_model = self.choose_best_model([model_1, model_2, model_3, model_4])

        return best_model


    def build_model_1(self, x_train, y_train, model_type):
        """
        This is for LightGBM Regressor and Classifier

        """
        if model_type =='regressor':
            model = LGBMRegressor(random_state=SEED)
        else:
            model = LGBMClassifier(random_state=SEED)

        model.fit(x_train, np.ravel(y_train))

        return model


    def build_model_2(self, x_train, y_train, model_type):
        """
        This is for KNN Regressor and Classifier

        """
        if model_type =='regressor':
            model = KNeighborsRegressor()
        else:
            model = KNeighborsClassifier()
 
        model.fit(x_train, np.ravel(y_train))

        return model


    def build_model_3(self, x_train, y_train, model_type):
        """
        This is for Linear Regressor and Linear Classifier

        """
        if model_type =='regressor':
            model = LinearRegression()
        else:
            model = LogisticRegression()
 
        model.fit(x_train, np.ravel(y_train))

        return model


    def build_model_4(self, x_train, y_train, model_type):
        """
        This is for Random Forest Regressor and Classifier

        """
        if model_type =='regressor':
            model = RandomForestRegressor(random_state=SEED)
        else:
            model = RandomForestClassifier(random_state=SEED)
 
        model.fit(x_train, np.ravel(y_train))

        return model


    def choose_best_model(self, mod_list):
        metrics = {}
        if self.model_type =='regressor':
            for model in mod_list:
                preds = model.predict(self.x_test)
                score = r2_score(self.y_test, preds)
                metrics.update({model:score})
            
        return metrics