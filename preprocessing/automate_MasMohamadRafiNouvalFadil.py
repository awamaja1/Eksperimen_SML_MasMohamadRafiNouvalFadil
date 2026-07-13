import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os

def preprocess_data(input_path: str, output_dir: str):
    print(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)

    # Simple EDA metrics print
    print("Data shape:", df.shape)
    print("Missing values before:\n", df.isnull().sum())

    # Preprocessing
    # 1. Handle missing values
    imputer = SimpleImputer(strategy='mean')
    features = [c for c in df.columns if c != 'target']
    
    df[features] = imputer.fit_transform(df[features])
    
    print("Missing values after:\n", df.isnull().sum())

    # 2. Split data
    X = df[features]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Reconstruct dataframes
    train_df = pd.DataFrame(X_train_scaled, columns=features)
    train_df['target'] = y_train.values

    test_df = pd.DataFrame(X_test_scaled, columns=features)
    test_df['target'] = y_test.values

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Preprocessed data saved to {output_dir}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_csv = os.path.join(base_dir, "credit_scoring_raw", "dataset.csv")
    output_directory = os.path.join(base_dir, "preprocessing", "credit_scoring_preprocessing")
    
    preprocess_data(input_csv, output_directory)
