from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor, early_stopping, log_evaluation


def random_forest_training(param_dict, X_train, y_train):
    model = RandomForestRegressor(**param_dict)
    model.fit(X_train, y_train)
    return model


def decision_tree_training(param_dict, X_train, y_train):
    model = DecisionTreeRegressor(**param_dict)
    model.fit(X_train, y_train)
    return model


def lgboost_training(param_dict, X_train, y_train, X_test, y_test, stopping_rounds=450):
    model = LGBMRegressor(
        **param_dict,
        n_jobs=-1,
        verbose=10,
        boosting_type="gbdt",
        objective="regression",
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="rmse",
        callbacks=[
            early_stopping(stopping_rounds=stopping_rounds, verbose=False),
            log_evaluation(10),
        ],
    )
    return model