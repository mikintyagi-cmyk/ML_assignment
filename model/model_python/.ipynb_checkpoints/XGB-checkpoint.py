import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
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



def xgb_classifier(X,y):
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=40
    )
    scaler = StandardScaler() # Scaling (VERY IMPORTANT for KNN)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # Create model
    model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
    )
    
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