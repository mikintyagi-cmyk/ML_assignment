import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)


def decision_classifier(X,y):
    
    numerical_cols = X.select_dtypes(include=["int64","float64"]).columns #finding numerical data to handle null value
    categorical_cols = X.select_dtypes(include=["object"]).columns #finding categorical data to handle null value
    
    # Train-test split 20% test 80% training and random sample is 50
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=40, stratify=y
    )
    
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, numerical_cols),
        ("cat", cat_pipeline, categorical_cols)
    ])
    
    
    model = Pipeline([
        ("prep", preprocessor),
        ("dt", DecisionTreeClassifier(
            max_depth=None,           # None = grow until pure leaves (control overfitting with other params)
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced",  # same as you used before
            random_state=42,          # for reproducibility
            criterion="gini"          # can also use "entropy"
        ))
    ])  
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    for t in [0.25, 0.3, 0.35, 0.4, 0.45]:
        y_pred = (y_prob >= t).astype(int)
    
    baseline_metrics ={
    "accuracy": accuracy_score(y_test, y_pred),
    "auc": float(roc_auc_score(y_test, y_prob)),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred),
    "mcc": float(matthews_corrcoef(y_test, y_pred))
    }

    return baseline_metrics, y_test, y_pred 
