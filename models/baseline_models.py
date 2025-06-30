from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
import warnings
from sklearn.exceptions import ConvergenceWarning

import numpy as np

warnings.filterwarnings("ignore", category=ConvergenceWarning)


alpha_grid = np.logspace(-6, 5, 45)

# Linear evaluation models
LogReg = ( # (pipeline, grid) for Logistic Regression classifier
    Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("varth", VarianceThreshold()), 
        ("scale", StandardScaler()),
        ("model_LR", LogisticRegression(max_iter=2000, class_weight='balanced'))
        ]),
    {"model_LR__C" : alpha_grid}, 
)

LinReg = ( # (pipeline, grid) for Ridge Regression
    Pipeline([
        ("varth", VarianceThreshold()), 
        ("scale", StandardScaler()),
        ("model_Ridge", Ridge(max_iter=2000))
        ]),
    {"model_Ridge__alpha" : alpha_grid}, 
)

# Multi-label
OVR_LogReg = ( # (pipeline, grid) for Logistic Regression classifier
    Pipeline([
        ("varth", VarianceThreshold()), 
        ("scale", StandardScaler()),
        ("model_OVR_LR", OneVsRestClassifier(LogisticRegression(max_iter=2000, class_weight="balanced")))
        ]),
    {"model_OVR_LR__estimator__C" : alpha_grid}, 
)

# Multi-class
SM_LogReg = (
    Pipeline([
        ("varth", VarianceThreshold()),
        ("scale", StandardScaler()),
        ("model_SM_LR", LogisticRegression(max_iter=2000, class_weight="balanced",
                                           #multi_class="multinomial", solver="lbfgs"))
                                           solver="lbfgs"))
    ]),
    {"model_SM_LR__C" : alpha_grid}, 
)