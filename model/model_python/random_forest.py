import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)



def rf_classifier(X,y):
    
    numerical_cols = X.select_dtypes(include=["int64","float64"]).columns #finding numerical data to handle null value
    categorical_cols = X.select_dtypes(include=["object"]).columns #finding categorical data to handle null value
    
    # Train-test split 20% test 80% training and random sample is 40
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
        ("lr", RandomForestClassifier( n_estimators=100, random_state=42, n_jobs=-1 ))
    ])

    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Step 10: Evaluation Metrics
    baseline_metrics ={
    "Accuracy": accuracy_score(y_test, y_pred),
    "AUC": float(roc_auc_score(y_test, y_prob)),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1": f1_score(y_test, y_pred),
    "MCC": float(matthews_corrcoef(y_test, y_pred))
    }
    return baseline_metrics, y_test, y_pred 
