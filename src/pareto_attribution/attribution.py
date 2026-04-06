import numpy as np
from lightgbm import LGBMRegressor
import shap

def train_model(X, y):
    model = LGBMRegressor(n_estimators=200)
    model.fit(X, y)
    return model

def compute_shap(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    importance = np.mean(np.abs(shap_values), axis=0)
    return shap_values, importance