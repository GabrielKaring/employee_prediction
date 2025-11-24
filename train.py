import pandas as pd
import mlflow
import mlflow.sklearn
import joblib  # Importante para salvar localmente
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score

# Configuração Opcional do MLflow (apenas para registro histórico)
# Se der erro de conexão, pode comentar as linhas de mlflow set_tracking_uri
try:
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("HR_Attrition_Prediction")
except:
    print("Aviso: MLflow server não encontrado, seguindo apenas com arquivo local.")

def load_data(path):
    df = pd.read_csv(path)
    df = df.drop(['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'], axis=1)
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    return df

def train():
    # Ajuste o caminho conforme sua estrutura de pastas
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

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
    ])

    # Treinamento
    print("Treinando o modelo...")
    pipeline.fit(X_train, y_train)

    # Métricas
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"AUC: {roc_auc_score(y_test, y_prob):.4f}")

    # --- SALVAMENTO DO ARQUIVO FÍSICO (CRUCIAL PARA O DOCKER) ---
    joblib.dump(pipeline, "model.pkl")
    print("\nSUCESSO: Arquivo 'model.pkl' salvo na pasta do projeto.")

    # Tentativa de log no MLflow (Opcional)
    try:
        with mlflow.start_run():
            mlflow.log_params({"model_type": "LogisticRegression"})
            mlflow.log_metrics({"roc_auc": roc_auc_score(y_test, y_prob)})
            mlflow.sklearn.log_model(pipeline, "model")
            print("Log no MLflow realizado com sucesso.")
    except Exception as e:
        print(f"Pulei o log do MLflow: {e}")

if __name__ == "__main__":
    train()