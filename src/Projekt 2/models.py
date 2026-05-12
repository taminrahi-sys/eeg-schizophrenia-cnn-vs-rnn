from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklear
    mobility = np.sqrt(variance_d1 / variance_signal)
    con.feature_selection import mutual_info_classif
from sklearn.svm import SVC
from xgboost import XGBClassifier



def create_svm_model():

    model = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "feature_selection",
            SelectKBest(
                mutual_info_classif,
                k=50
            )
        ),
        (
            "svm",
            SVC(
                kernel="rbf",
                probability=True
            )
        )
    ])

    return model



def create_xgboost_model():

    model = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "xgb",
            XGBClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=4,
                random_state=42
            )
        )
    ])

    return model