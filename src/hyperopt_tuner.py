from hyperopt import fmin, tpe, hp, STATUS_OK


def objective(args):
    
    # Initialize model pipeline
    pipe = Pipeline(steps=[('model', args['model'])])
    
    pipe.set_params(**args['params']) # Model parameters will be set here
    
    # Cross Validation Score. Note the transformer.fit_transform for X_train. 
    
    score = cross_val_score(pipe, transformer.fit_transform(X_train), y_train, cv=5, n_jobs=-1, error_score=0.99)
    print(f'Model Name: {args['model']}: ', score)
          
    # Since we have to minimize the score, we return 1- score.
    return {'loss': 1 - np.median(score), 'status': STATUS_OK}


# Defining Search Space
space = hp.choice('classifiers', [
    {
    'model':KNeighborsClassifier(),
    'params':{
        'model__n_neighbors': hp.choice('knc.n_neighbors', range(2,10)),
        'model__algorithm': hp.choice('knc.algorithm',
                                      ['auto', 'ball_tree', 'kd_tree']),
        'model__metric': hp.choice('knc.metric', ['chebyshev', 'minkowski'])
    }
    },
    {
    'model':SVC(),
    'params':{
        'model__C': hp.choice('C', np.arange(0.005,1.0,0.01)),
        'model__kernel': hp.choice('kernel',['linear', 'rbf', 'sigmoid']),
        'model__degree':hp.choice('degree',[2,3,4]),
        'model__gamma': hp.uniform('gamma',0.001,1000)
    }
    },

    {
    'model': LogisticRegression(verbose=0),
    'params': {
        'model__penalty': hp.choice('lr.penalty', ['none', 'l2']),
        'model__C': hp.choice('lr.C', np.arange(0.005,1.0,0.01))

    }
    },
        {
    'model': XGBClassifier(eval_metric='logloss', verbosity=0),
    'params': {
        'model__max_depth' : hp.choice('xgb.max_depth',
                                       range(5, 30, 1)),
        'model__learning_rate' : hp.quniform('xgb.learning_rate',
                                             0.01, 0.5, 0.01),
        'model__n_estimators' : hp.choice('xgb.n_estimators',
                                          range(5, 50, 1)),
        'model__reg_lambda' : hp.uniform ('xgb.reg_lambda', 0,1),
        'model__reg_alpha' : hp.uniform ('xgb.reg_alpha', 0,1)
    }
    },
    {
        'model': QuadraticDiscriminantAnalysis(), # Default params
        'params': {}
    }
])


# Hyperopts Trials() records all the model and run artifacts.
trials = Trials()

# Fmin will call the objective funbction with selective param set. 
# The choice of algorithm will narrow the searchspace.

best_classifier = fmin(objective, space, algo=tpe.suggest,
                       max_evals=50, trials=trials)

# Best_params of the best model
best_params = space_eval(space, best_classifier)