import joblib
import pandas as pd
import os

class AttritionPredictor:
    def __init__(self):
        model_path = "model.pkl"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"ERRO CRÍTICO: O arquivo '{model_path}' não foi encontrado. "
                "Execute 'python train.py' primeiro para treinar e escolher o melhor modelo."
            )
            
        print(f"Carregando modelo (pipeline) de: {model_path}")
        self.model = joblib.load(model_path)

    def predict(self, data: dict):
        # Converte dicionário para DataFrame
        df_input = pd.DataFrame([data])
        
        # O pipeline cuida do StandardScaler automaticamente se necessário
        prediction = self.model.predict(df_input)
        probability = self.model.predict_proba(df_input)[:, 1]
        
        status = "Saiu" if prediction[0] == 1 else "Ficou"
        
        return {
            "prediction": int(prediction[0]),
            "probability": float(probability[0]),
            "status": status,
            "risk_level": "Alto" if probability[0] > 0.6 else "Médio" if probability[0] > 0.4 else "Baixo"
        }

if __name__ == "__main__":
    # Teste rápido
    sample = {
        "DistanceFromHome": 10, "EnvironmentSatisfaction": 1, "JobSatisfaction": 1,
        "YearsAtCompany": 2, "WorkLifeBalance": 1, "PercentSalaryHike": 11,
        "YearsInCurrentRole": 1, "YearsSinceLastPromotion": 0
    }
    try:
        p = AttritionPredictor()
        print(p.predict(sample))
    except Exception as e:
        print(e)