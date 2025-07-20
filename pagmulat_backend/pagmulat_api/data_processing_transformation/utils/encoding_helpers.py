import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

def one_hot_encode(df, column, prefix):
    return pd.get_dummies(df[column], prefix=prefix)

def multilabel_binarize(df, column, prefix):
    mlb = MultiLabelBinarizer()
    transformed = mlb.fit_transform(df[column])
    return pd.DataFrame(transformed, columns=[f"{prefix}_{cls}" for cls in mlb.classes_])
