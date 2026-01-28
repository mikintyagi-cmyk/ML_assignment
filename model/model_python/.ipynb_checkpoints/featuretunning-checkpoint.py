import pandas as pd
from sklearn.preprocessing import LabelEncoder


def dataprocess(df, target_column=None):
    df = df.copy()

    drop_cols = [
        c for c in df.columns
        if c.lower() in ["id", "no", "index", "sno", "srno"]
           or c.lower().startswith("unnamed")]

    df.drop(columns=drop_cols, inplace=True)
    if target_column is None:

        # Common target names
        common_targets = ["income"]
        for col in df.columns:
            if col.lower() in common_targets:
                target_column = col
                break

        # If still not found → assume last column
        if target_column is None:
            target_column = df.columns[-1]
            
    X = df.drop(columns=[target_column])
    y = df[target_column]

    if y.dtype == "object":
        y = LabelEncoder().fit_transform(y)

    X = pd.get_dummies(X, drop_first=True)

    return X, y, target_column