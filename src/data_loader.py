import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(filepath="../heart+disease/processed.cleveland.data"):
    columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
    ]
    df = pd.read_csv(filepath, names=columns, na_values="?")
    df = df.dropna()
    
    # Binary classification map
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    # 75/25 split, stratified
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=100, stratify=y
    )
    
    return df, X_train, X_test, y_train, y_test

def get_clinical_tiers():
    return {
        "Tier 1: Baseline": ["age", "sex", "cp"],
        "Tier 2: Vitals": ["age", "sex", "cp", "trestbps", "fbs"],
        "Tier 3: Blood Lab": ["age", "sex", "cp", "trestbps", "fbs", "chol"],
        "Tier 4: Stress Test": ["age", "sex", "cp", "trestbps", "fbs", "chol", 
                                "restecg", "thalach", "exang", "oldpeak", "slope"],
        "Tier 5: Specialized": ["age", "sex", "cp", "trestbps", "fbs", "chol", 
                                 "restecg", "thalach", "exang", "oldpeak", "slope", 
                                 "ca", "thal"]
    }