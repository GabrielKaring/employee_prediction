import joblib
import pandas as pd
import os

class AttritionPredictor:
    def __init__(self):
        # O arquivo deve estar na mesma pasta que este script
        model_path = "model.pkl"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"ERRO CRÍTICO: O arquivo '{model_path}' não foi encontrado. "
                "Você rodou o 'python train.py' antes de construir o Docker?"
            )
            
        print(f"Carregando modelo local de: {model_path}")
        self.model = joblib.load(model_path)

    def predict(self, data: dict):
        # Converte dicionário para DataFrame
        df_input = pd.DataFrame([data])
        
        # Faz a predição
        prediction = self.model.predict(df_input)
        probability = self.model.predict_proba(df_input)[:, 1]
        
        return {
            "prediction": int(prediction[0]),
            "probability": float(probability[0]),
            "status": "Saiu" if prediction[0] == 1 else "Ficou"
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