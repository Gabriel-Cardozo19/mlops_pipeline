import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def load_data(path):
    df = pd.read_csv(path)
    return df



def feature_engineering(df):

    # -------- TARGET --------
    y = df["Pago_atiempo"]
    X = df.drop(columns=["Pago_atiempo"])

    X = X.drop(columns=["puntaje","puntaje_datacredito"])

    # -------- COLUMN TYPES --------
    numeric_features = X.select_dtypes(include=["int64","float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object","category"]).columns.tolist()

    # ejemplo ordinal (si aplica)
    ordinal_features = []

    # -------- PIPELINES --------

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    ordinal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder())
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
        ("ord", ordinal_pipeline, ordinal_features)
    ])


    # -------- SPLIT --------

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, preprocessor
