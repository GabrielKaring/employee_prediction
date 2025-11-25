import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score

# Configuração para salvar logs LOCALMENTE (evita erro de conexão)
mlflow.set_experiment("HR_Attrition_MultiModel")

def load_data(path):
    df = pd.read_csv(path)
    df = df.drop(['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'], axis=1)
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    return df

def train_and_compare():
    data_path = 'data/WA_Fn-UseC_-HR-Employee-Attrition.csv'
    
    try:
        df = load_data(data_path)
    except FileNotFoundError:
        print(f"ERRO: Arquivo não encontrado em {data_path}")
        return

    features = [
        "DistanceFromHome", "EnvironmentSatisfaction", "JobSatisfaction",
        "YearsAtCompany", "WorkLifeBalance", "PercentSalaryHike",
        "YearsInCurrentRole", "YearsSinceLastPromotion"
    ]

    X = df[features]
    y = df['Attrition']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    models_to_train = {
        'Regressão Logística': Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
        ]),
        'Árvore de Decisão': Pipeline([
            ('scaler', StandardScaler()), 
            ('model', DecisionTreeClassifier(random_state=42, class_weight='balanced'))
        ]),
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestClassifier(random_state=42, class_weight='balanced'))
        ])
    }

    best_model_pipeline = None
    best_recall = -1
    best_model_name = ""

    print("\n--- INICIANDO TREINAMENTO ---")

    for name, pipeline in models_to_train.items():
        with mlflow.start_run(run_name=name, nested=True):
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            rec = recall_score(y_test, y_pred)
            
            print(f"Modelo: {name} | Recall: {rec:.4f}")
            
            mlflow.log_params({"model_name": name})
            mlflow.log_metrics({"recall": rec})
            mlflow.sklearn.log_model(pipeline, "model")

            if rec > best_recall:
                best_recall = rec
                best_model_pipeline = pipeline
                best_model_name = name

    print(f"\nVENCEDOR: {best_model_name}")
    joblib.dump(best_model_pipeline, "model.pkl")
    print(f"SUCESSO: Modelo salvo como 'model.pkl'.")

if __name__ == "__main__":
    train_and_compare()