from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.metrics import mean_squared_error, r2_score, f1_score
import pandas as pd
import numpy as np
from config import SEED, MODEL_PICKLE_PATH
from hyper_tuner import HyperTuner
import pickle as pkl
import logging

class Modeller:

    def __init__(self, x_train, y_train, x_test, y_test, model_type, num_cat_cols, feature_select=False):
        """
        Class Modeller constructor

        Accepts the inputs train and test datasets
        and model_type and returns the best model suitable for the given dataset
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.model_type = model_type
        self.num_cat_cols = num_cat_cols
        self.feature_select = feature_select
        self.ht = HyperTuner()


    def build_model(self):
        """
        The orchestrator which tests multiple hypothesises. This method
        uses member methods to build different models and returns the metrics related to
        each experimented model
        """
        # implement feature selection code here
        # this should return selected features list as well

        x_train = self.x_train.copy()
        y_train =  self.y_train.copy()

        if self.feature_select:
            selected_feat_list = self.get_important_features(self.x_train, self.y_train, self.model_type)

            print("Selected Features: ",selected_feat_list)

            x_train = x_train[selected_feat_list]
            self.x_test = self.x_test[selected_feat_list]

        # save x_train, x_test here , saving x_test_df to verify it in flask UI
        # The same datatsets can also be used to train hyperopt
        x_train.to_csv(MODEL_PICKLE_PATH + "model_train_x_df.csv", index=False, header=True)
        y_train.to_csv(MODEL_PICKLE_PATH + "model_train_y_df.csv", index=False, header=True)
        self.x_test.to_csv(MODEL_PICKLE_PATH + "model_test_x_df.csv", index=False, header=True)


        #build model 1 - LGBM
        model_1, best_params = self.build_model_1(x_train, y_train, self.model_type)

        #build model 2 - KNN
        model_2, best_params = self.build_model_2(x_train, y_train, self.model_type)

        #build model 3 - Linear/Logistic
        model_3, best_params = self.build_model_3(x_train, y_train, self.model_type)

        #build model 4 - RF
        model_4, best_params = self.build_model_4(x_train, y_train, self.model_type)

        #build model 5 - SVM
        model_5, best_params = self.build_model_5(x_train, y_train, self.model_type)

        #build model 6 - Ridge
        model_6, best_params = self.build_model_6(x_train, y_train, self.model_type)

        best_model = self.choose_best_model([model_1, model_2, model_3, model_4, model_5, model_6])

        return best_model


    def get_important_features(self, x_train_df, y_train_df, model_type):
        """
        This function implements feature selction
        to reduce dimensionality. This is optional
        """
        print("Feature Selection is Running.....")

        if model_type =='regressor':
            model = RandomForestRegressor(random_state=SEED)

        else:
            model = RandomForestClassifier(random_state=SEED)

        feat_est =  RFECV(model)
        feat_est.fit(x_train_df, np.ravel(y_train_df))

        sup_indices = feat_est.get_support(indices=True)
        
        # write code get feature names from the masked array
        selected_feat = [x_train_df.columns.tolist()[i] for i in sup_indices]

        return selected_feat


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

        else:
            model = LGBMClassifier(random_state=SEED)

            # as of now implement random grid for hyper-param tuning
            param_grid = {
                'boosting_type': ['gbdt', 'dart'],
                'num_leaves': [7, 14, 21, 28, 31, 50],
                'learning_rate': [0.1, 0.03, 0.003],
                'max_depth': [-1, 3, 5],
                'n_estimators': [50, 100, 200, 500],
                'class_weight': ['balanced', None]
                }

            best_model, best_params = self.ht.get_best_model(model, param_grid, x_train, np.ravel(y_train))

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

        else:
            model = KNeighborsClassifier()

            # as of now implement random grid for hyper-param tuning
            param_grid = {
                'n_neighbors': [3, 5, 8],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'leaf_size': [20,30,40,50]
                }

            best_model, best_params = self.ht.get_best_model(model, param_grid, x_train, np.ravel(y_train))
 
        return best_model, best_params


    def build_model_3(self, x_train, y_train, model_type):
        """
        This is for Linear Regressor and Linear Classifier

        """
        if model_type =='regressor':
            model = LinearRegression()
            param_grid = {
            'fit_intercept': [True, False]
            }
            #best_model.fit(x_train, np.ravel(y_train))
            best_model, best_params = self.ht.get_best_model(model, param_grid, x_train, np.ravel(y_train))

        else:
            model = LogisticRegression(random_state=SEED)
            param_grid = {
                'C': [0.2, 0.4, 0.6, 1],
                'class_weight': ['balanced', None],
                'solver': ['newton-cg', 'lbfgs', 'liblinear']
                }

            best_model, best_params = self.ht.get_best_model(model, param_grid, x_train, np.ravel(y_train))
 
        return best_model, best_params


    def build_model_4(self, x_train, y_train, model_type):
        """
        This is for Random Forest Regressor and Classifier

        """
        if model_type =='regressor':
            model = RandomForestRegressor(random_state=SEED)

            # as of now implement random grid for hyper-param tuning
            param_grid = {
                'criterion' : ['squared_error','poisson'],
                'max_depth': [3, 5],
                'n_estimators': [50, 100, 200, 500],
                'min_samples_split' : [2]
                }

            best_model, best_params = self.ht.get_best_model(model, param_grid, x_train, np.ravel(y_train))

        else:
            model = RandomForestClassifier(random_state=SEED)

            # as of now implement random grid for hyper-param tuning
            param_grid = {
                'criterion' : ['gini','entropy'],
                'max_depth':  [3, 5],
                'n_estimators': [50, 100, 200, 500]
                }

            best_model, best_params = self.ht.get_best_model(model, param_grid, x_train, np.ravel(y_train))

        return best_model, best_params


    def build_model_5(self, x_train, y_train, model_type):
        """
        This is for Support Vector Machine Regressor and Classifier

        """
        if model_type =='regressor':
            model = SVR()

            # as of now implement random grid for hyper-param tuning
            param_grid = {
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'degree': [2,3,4],
                'C': [1,2]
                }

            best_model, best_params = self.ht.get_best_model(model, param_grid, x_train, np.ravel(y_train))

        else:
            model = SVC()

            # as of now implement random grid for hyper-param tuning
            param_grid = {
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'degree': [2,3,4],
                'C': [1,2],
                'class_weight': ['balanced', None],
                'random_state': [SEED, None]
                }

            best_model, best_params = self.ht.get_best_model(model, param_grid, x_train, np.ravel(y_train))

        return best_model, best_params


    def build_model_6(self, x_train, y_train, model_type):
        """
        This is for Ridge Regressor and Classifer

        """
        if model_type =='regressor':
            model = Ridge()

            # as of now implement random grid for hyper-param tuning
            param_grid = {
                'alpha': [0.2, 0.4, 0.6, 1],
                'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs'],
                'random_state': [SEED, None]
                }

            best_model, best_params = self.ht.get_best_model(model, param_grid, x_train, np.ravel(y_train))

        else:
            model = RidgeClassifier()

            # as of now implement random grid for hyper-param tuning
            param_grid = {
                'alpha': [0.2, 0.4, 0.6, 1],
                'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs'],
                'random_state': [SEED, None],
                'class_weight': ['balanced', None]
                }

            best_model, best_params = self.ht.get_best_model(model, param_grid, x_train, np.ravel(y_train))

        return best_model, best_params


    def choose_best_model(self, mod_list):
        """
        Currently saves the trained model pickles and returns the
        metrics related to all the best models chosen by HyperTuner class
        """
        metrics = {}
        if self.model_type =='regressor':
            for idx, model in enumerate(mod_list):
                preds = model.predict(self.x_test)
                score = r2_score(self.y_test, preds)
                metrics.update({"model_"+str(idx+1):{model:score}})
                with open(MODEL_PICKLE_PATH + "model_"+str(idx+1)+".pkl", "wb") as handle:
                    pkl.dump(model, handle)
        else:
            for idx, model in enumerate(mod_list):
                preds = model.predict(self.x_test)
                pos_label = preds.tolist()[1]
                score = f1_score(self.y_test, preds, pos_label=pos_label)
                metrics.update({"model_"+str(idx+1):{model:score}})
                with open(MODEL_PICKLE_PATH + "model_"+str(idx+1)+".pkl", "wb") as handle:
                    pkl.dump(model, handle)
            
        return metrics