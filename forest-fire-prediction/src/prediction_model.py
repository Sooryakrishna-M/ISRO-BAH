# src/prediction_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model():
    df = pd.read_csv('data/processed/cleaned_data.csv')
    X = df[['brightness', 'frp']]
    y = df['high_risk']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    joblib.dump(model, 'src/fire_model.pkl')
    print("✅ Model trained and saved.")

if __name__ == "__main__":
    train_model()
