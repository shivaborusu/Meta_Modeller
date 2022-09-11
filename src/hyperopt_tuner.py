import pandas as pd
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from config import SEED, DATA_SET_PATH, MODEL_PICKLE_PATH
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
from config import SEED, MODEL_PICKLE_PATH
import pickle as pkl
import logging


model_type = 'regresson'

space_reg = hp.choice('regressors', [
    {
        'model': LGBMRegressor(random_state=SEED),
        'params': {
            'num_leaves': hp.choice('num_leaves', [7, 14, 21, 28, 31, 50]),
            'learning_rate': hp.choice('learning_rate', [0.1, 0.03, 0.003]),
            'max_depth': hp.choice('max_depth', [-1, 3, 5]),
            'n_estimators': hp.choice('n_estimators', [50, 100, 200, 500])
        },
        'metric' : "r2"
    },
    {
        'model': KNeighborsRegressor(),
        'params': {
            'n_neighbors': hp.choice('n_neighbors', [3, 5, 8]),
            'weights': hp.choice('weights', ['uniform', 'distance']),
            'algorithm': hp.choice('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])
        },
        'metric' : "r2"
    },
    {
        'model': LinearRegression(),
        'params': {},
        'metric' : "r2"
    },
    {
        'model': SVR(),
        'params': {
            'kernel': hp.choice('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
            'degree': hp.choice('degree', [2,3,4]),
            'C': hp.choice('C', [1, 2])
        },
        'metric': "r2"
    },
    {
        'model': Ridge(),
        'params': {
            'alpha': hp.choice('alpha', [0.2, 0.4, 0.6, 1]),
            'solver': hp.choice('solver', ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs']),
            'random_state': hp.choice('random_state', [SEED, None])
        },
        'metric': "r2"
    }
    # due to this issue RandomForest is avoided 
    # https://github.com/hyperopt/hyperopt/issues/823
    # {
    #     'model': RandomForestRegressor(random_state=SEED),
    #     'params': {
    #         'criterion': hp.choice('criterion', ['squared_error','poisson']),
    #         'max_depth': hp.choice('max_depth', [3, 5]),
    #         'n_estimators': hp.choice('n_estimators', [50, 100, 200, 500])
    #     }
    # },
    ])

# classifier grid definition is still pending
space_clf = hp.choice('classifiers', [
    {
        'model': LGBMClassifier(random_state=SEED),
        'params': {
            'num_leaves': hp.choice('num_leaves', [7, 14, 21, 28, 31, 50]),
            'learning_rate': hp.choice('learning_rate', [0.1, 0.03, 0.003]),
            'max_depth': hp.choice('max_depth', [-1, 3, 5]),
            'n_estimators': hp.choice('n_estimators', [50, 100, 200, 500]),
            'boosting_type': hp.choice('boosting_type', ['gbdt', 'dart']),
            'class_weight': hp.choice('class_weight', ['balanced', None]),
        },
        'metric' : "f1"
    },
    {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': hp.choice('n_neighbors', [3, 5, 8]),
            'weights': hp.choice('weights', ['uniform', 'distance']),
            'algorithm': hp.choice('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']),
            'leaf_size': hp.choice('leaf_size', [20,30,40,50])
        },
        'metric' : "f1"
    },
    {
        'model': LogisticRegression(),
        'params': {''},
        'metric' : "f1"
    },
    {
        'model': SVR(),
        'params': {
            'kernel': hp.choice('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
            'degree': hp.choice('degree', [2,3,4]),
            'C': hp.choice('C', [1, 2])
        },
        'metric': "r2"
    },
    {
        'model': Ridge(),
        'params': {
            'alpha': hp.choice('alpha', [0.2, 0.4, 0.6, 1]),
            'solver': hp.choice('solver', ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs']),
            'random_state': hp.choice('random_state', [SEED, None])
        },
        'metric': "r2"
    }
])

def hyperparameter_tuning(space):
    x_train = pd.read_csv(MODEL_PICKLE_PATH + "model_train_x_df.csv", header='infer')
    y_train = pd.read_csv(MODEL_PICKLE_PATH + "model_train_y_df.csv", header='infer')

    model = space['model']
    reg = model.set_params(**space['params'])
    metric = space['metric']

    acc = cross_val_score(reg, x_train, np.ravel(y_train), scoring=metric).mean()
    return {"loss": -acc, "status": STATUS_OK}


# Initialize trials object
trials = Trials()

if model_type == 'regresson':
    space = space_reg
else:
    space = space_clf

best = fmin(
    fn=hyperparameter_tuning,
    space = space, 
    algo=tpe.suggest, 
    max_evals=10,
    trials=trials,
    return_argmin=True
)

print("Best: {}".format(best))

# Best_params of the best model
print(space_eval(space_reg, best))