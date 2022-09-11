import numpy as np
import pandas as pd
from config import SEED
from sklearn.model_selection import RandomizedSearchCV


class HyperTuner:
    def __init__(self):
        self.count = 0

    def get_best_model(self, model, param_grid, x_train, y_train):
        model = RandomizedSearchCV(estimator = model, param_distributions=param_grid, random_state=SEED, cv=5, error_score='raise', n_jobs=-1)
        search = model.fit(x_train, y_train)
        
        self.count+=1
        print(f"Best Model for Model ID {str(self.count)}:   ", search.best_estimator_)
        print(f"Best Params for Model ID {str(self.count)}:   ", search.best_params_, end='\n')
        print("")

        return search.best_estimator_, search.best_params_


