from hyperopt import fmin, tpe, hp, STATUS_OK


def objective(params):
    clf = RandomForestClassifier(**params,n_jobs=-1)
    acc = cross_val_score(clf, X_scaled, y,scoring="accuracy").mean()
    return {"loss": -acc, "status": STATUS_OK}



space = {
    "n_estimators": hp.choice("n_estimators", [100, 200, 300, 400,500,600]),
    "max_depth": hp.quniform("max_depth", 1, 15,1),
    "criterion": hp.choice("criterion", ["gini", "entropy"]),
}


space = hp.uniform('x', -10, 10)

best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100)

print(best)