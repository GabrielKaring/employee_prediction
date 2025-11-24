from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from predict import AttritionPredictor
import uvicorn

app = FastAPI(title="HR Attrition Prediction API", version="1.0")

# Inicializa o preditor ao iniciar a API
predictor = None
try:
    predictor = AttritionPredictor()
except Exception as e:
    print(f"ERRO DE INICIALIZAÇÃO: {e}")

# Definição do Schema de Entrada
class EmployeeData(BaseModel):
    DistanceFromHome: int
    EnvironmentSatisfaction: int
    JobSatisfaction: int
    YearsAtCompany: int
    WorkLifeBalance: int
    PercentSalaryHike: int
    YearsInCurrentRole: int
    YearsSinceLastPromotion: int

@app.get("/")
def health_check():
    if predictor:
        return {"status": "API is running", "model_loaded": True}
    return {"status": "API is running", "model_loaded": False, "error": "Model not found"}

@app.post("/predict")
def predict_attrition(data: EmployeeData):
    if not predictor:
        raise HTTPException(
            status_code=500, 
            detail="Modelo não carregado no servidor. Verifique se 'model.pkl' existe."
        )
    
    try:
        input_data = data.dict()
        result = predictor.predict(input_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)