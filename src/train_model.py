import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def train_random_forest_in_chunks(X_train, y_train):
    print("X_train NaN değer sayısı:", np.isnan(X_train).sum().sum())
    print("X_train sonsuz değer sayısı:", np.isinf(X_train).sum().sum())
    
    print("y_train tipi:", type(y_train))
    
    if isinstance(y_train, pd.Series):
        print("y_train NaN değer sayısı:", y_train.isna().sum())
        y_train = y_train.dropna() 
        X_train = X_train.loc[y_train.index] 
    else:
        print("y_train NaN değer sayısı:", np.isnan(y_train).sum())

    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    
    return rf_model
