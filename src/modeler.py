from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
from config import SEED, MODEL_PICKLE_PATH
from hyper_tuner import HyperTuner
import pickle as pkl

class Modeller:
    def __init__(self, x_train, y_train, x_test, y_test, model_type):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.model_type = model_type
        self.ht = HyperTuner()


    def build_model(self):
        # implement feature selection code here
        # this should return selected features list as well



        #build model 1 - LGBM
        model_1, best_params = self.build_model_1(self.x_train, self.y_train, self.model_type)

        #build model 1
        model_2, best_params = self.build_model_2(self.x_train, self.y_train, self.model_type)

        #build model 1
        model_3, best_params = self.build_model_3(self.x_train, self.y_train, self.model_type)

        #build model 1
        model_4, best_params = self.build_model_4(self.x_train, self.y_train, self.model_type)

        best_model = self.choose_best_model([model_1, model_2, model_3, model_4])

        return best_model


    def build_model_1(self, x_train, y_train, model_type):
        """
        This is for LightGBM Regressor and Classifier

        """
        if model_type =='regressor':
            model = LGBMRegressor(random_state=SEED)

            # as of now implement random grid for hyper-param tuning
            param_grid = {
                'num_leaves': [7, 14, 21, 28, 31, 50],
                'learning_rate': [0.1, 0.03, 0.003],
                'max_depth': [-1, 3, 5],
                'n_estimators': [50, 100, 200, 500]
                }

            best_model, best_params = self.ht.get_best_model(model, param_grid, x_train, np.ravel(y_train))

            print("BEST_PARAMS LGBM: ", best_params)

        else:
            model = LGBMClassifier(random_state=SEED)

        return best_model, best_params


    def build_model_2(self, x_train, y_train, model_type):
        """
        This is for KNN Regressor and Classifier

        """
        if model_type =='regressor':
            model = KNeighborsRegressor()

            # as of now implement random grid for hyper-param tuning
            param_grid = {
                'n_neighbors': [3, 5, 8],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                }

            best_model, best_params = self.ht.get_best_model(model, param_grid, x_train, np.ravel(y_train))

            print("BEST_PARAMS KNN: ", best_params)

        else:
            model = KNeighborsClassifier()
 
        return best_model, best_params


    def build_model_3(self, x_train, y_train, model_type):
        """
        This is for Linear Regressor and Linear Classifier

        """
        if model_type =='regressor':
            model = LinearRegression()
        else:
            model = LogisticRegression()
 
        model.fit(x_train, np.ravel(y_train))
        best_params = {}

        return model, best_params


    def build_model_4(self, x_train, y_train, model_type):
        """
        This is for Random Forest Regressor and Classifier

        """
        if model_type =='regressor':
            model = RandomForestRegressor(random_state=SEED)

            # as of now implement random grid for hyper-param tuning
            param_grid = {
                'criterion' : ['squared_error','poisson'],
                'max_depth': [-1, 3, 5],
                'n_estimators': [50, 100, 200, 500]
                }

            best_model, best_params = self.ht.get_best_model(model, param_grid, x_train, np.ravel(y_train))

            print("BEST_PARAMS RF: ", best_params)
        else:
            model = RandomForestClassifier(random_state=SEED)

        return best_model, best_params


    def choose_best_model(self, mod_list):
        metrics = {}
        if self.model_type =='regressor':
            for idx, model in enumerate(mod_list):
                preds = model.predict(self.x_test)
                score = r2_score(self.y_test, preds)
                metrics.update({model:score})
                with open(MODEL_PICKLE_PATH + "model_"+str(idx)+".pkl", "wb") as handle:
                    pkl.dump(model, handle)
            
        return metrics