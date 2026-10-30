import pandas as pd
import numpy as np
import os
os.environ['USE_PYGEOS'] = '0'
from itertools import product
import math
from sklearn.ensemble import RandomForestRegressor 
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

def random_forest_training(param_dict, X_train, y_train):
    rf = RandomForestRegressor(**param_dict)
    rf.fit(X_train, y_train)
    return rf 


def decision_tree_training(param_dict, X_train, y_train):
    dt = DecisionTreeRegressor(**param_dict)
    dt.fit(X_train, y_train)
    return dt 


def lgboost_training(param_dict, X_train, y_train, X_test, y_test, stopping_rounds=450):
    # Using gpu for faster training
    lgbm_reg = LGBMRegressor(
        **param_dict, 
        # device='gpu', 
        # gpu_use_dp=True, 
        n_jobs=-1, 
        verbose=10, 
        boosting_type='gbdt',
        objective='regression'
        )
    # Stopping rounds to manually stop training
    lgbm_reg.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='rmse',
        callbacks=[early_stopping(stopping_rounds=stopping_rounds, verbose= False), log_evaluation(10)]
    )
    return lgbm_reg