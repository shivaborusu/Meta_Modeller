import numpy as np
import pandas as pd
from config import SEED
from sklearn.model_selection import RandomizedSearchCV


class HyperTuner:
    def __init__(self):
        pass

    def get_best_model(self, model, param_grid, x_train, y_train):
        model = RandomizedSearchCV(estimator = model, param_distributions=param_grid, random_state=SEED, cv=5, error_score='raise')
        search = model.fit(x_train, y_train)

        print("BEST_MODEL:   ", search.best_estimator_)

        return search.best_estimator_, search.best_params_

    def get_param_grid(self, model_type, model_version):
        pass

    def run_cross_validation(self):
        pass


