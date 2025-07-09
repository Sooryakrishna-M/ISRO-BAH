import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

# 1. Load and combine all CSV files
def load_all_data(folder="data/raw/MODIS"):
    files = glob.glob(os.path.join(folder, "*.csv"))
    df_list = [pd.read_csv(f) for f in files]
    return pd.concat(df_list, ignore_index=True)

# 2. Preprocess
def preprocess(df):
    df = df[['brightness', 'frp', 'bright_t31', 'confidence']].dropna()
    df['high_risk'] = df['confidence'].apply(lambda x: 1 if x >= 70 else 0)
    return df[['brightness', 'frp', 'bright_t31']], df['high_risk']

# 3. Train
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("🔍 Classification Report:\n", classification_report(y_test, y_pred))
    joblib.dump(model, "src/forest_fire_model_all_years.pkl")
    print("✅ Model saved to src/forest_fire_model_all_years.pkl")

if __name__ == "__main__":
    df_all = load_all_data()
    X, y = preprocess(df_all)
    train_model(X, y)
