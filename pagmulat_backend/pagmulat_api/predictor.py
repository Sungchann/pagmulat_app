import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_productivity_model():
    """Train model to predict productivity"""
    df = pd.read_excel("ModifiedFinalData.xlsx")
    
    # Prepare data
    X = df.drop(columns=['Productive_Yes', 'Productive_No', 'Productive_Sometimes'])
    y = df['Productive_Yes']
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    # Save model
    joblib.dump(model, 'productivity_model.pkl')
    return model.score(X_test, y_test)