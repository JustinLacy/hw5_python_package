import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from joblib import dump

from hw5_python_package.fingerprints import smiles_to_maccs, smiles_to_morgan

def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    if 'smiles' not in df.columns:
        for alt in ['SMILES', 'smile', 'Smile']:
            if alt in df.columns:
                df = df.rename(columns={alt: 'smiles'})
                break
    possible_y = ['exp', 'logP', 'measured_logP', 'value', 'Target']
    ycol = None
    for c in possible_y:
        if c in df.columns:
            ycol = c
            break
    if ycol is None:
        ycol = df.columns[-1]
    return df[['smiles', ycol]].rename(columns={ycol: 'y'})

def train_and_eval(X_train, X_test, y_train, y_test):
    model = MLPRegressor(hidden_layer_sizes=(100,100), max_iter=500, random_state=42)
    model.fit(X_train, y_train)
    ypred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, ypred))
    return model, rmse

def main():
    csv_path = os.path.join("data", "Lipophilicity.csv")
    if not os.path.exists(csv_path):
        print(f"ERROR: Can't find {csv_path}. Place your CSV there and try again.")
        sys.exit(1)

    df = load_data(csv_path)
    smiles = df['smiles'].astype(str).tolist()
    y = df['y'].astype(float).values

    sm_train, sm_test, y_train, y_test = train_test_split(smiles, y, test_size=0.2, random_state=42)

    print("Generating MACCS fingerprints...")
    X_train_maccs = smiles_to_maccs(sm_train)
    X_test_maccs  = smiles_to_maccs(sm_test)

    print("Generating Morgan fingerprints...")
    X_train_morgan = smiles_to_morgan(sm_train, radius=2, n_bits=2048)
    X_test_morgan  = smiles_to_morgan(sm_test,  radius=2, n_bits=2048)

    scaler = StandardScaler()
    y_train_scaled = scaler.fit_transform(y_train.reshape(-1,1)).ravel()

    print("Training MACCS MLPRegressor...")
    maccs_model, maccs_rmse = train_and_eval(X_train_maccs, X_test_maccs, y_train_scaled, scaler.transform(y_test.reshape(-1,1)).ravel())

    print("Training Morgan MLPRegressor...")
    morgan_model, morgan_rmse = train_and_eval(X_train_morgan, X_test_morgan, y_train_scaled, scaler.transform(y_test.reshape(-1,1)).ravel())

    os.makedirs("models", exist_ok=True)
    dump(maccs_model, "models/maccs_mlp.joblib")
    dump(morgan_model, "models/morgan_mlp.joblib")

    conda_env = os.getenv("CONDA_DEFAULT_ENV") or os.getenv("VIRTUAL_ENV") or "unknown-environment"

    print("\nRESULTS:")
    print(f"MACCS RMSE: {maccs_rmse:.4f}")
    print(f"Morgan RMSE: {morgan_rmse:.4f}")
    print(f"Conda environment: {conda_env}")

if __name__ == "__main__":
    main()
