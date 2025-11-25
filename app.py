from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from predict import AttritionPredictor
import uvicorn

app = FastAPI(title="HR Attrition Prediction API", version="2.0")

predictor = None
try:
    predictor = AttritionPredictor()
except Exception as e:
    print(f"ERRO DE INICIALIZAÇÃO: {e}")

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
    return {"status": "API online", "model_loaded": predictor is not None}

@app.post("/predict")
def predict_attrition(data: EmployeeData):
    if not predictor:
        raise HTTPException(status_code=500, detail="Modelo não carregado.")
    
    try:
        result = predictor.predict(data.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)