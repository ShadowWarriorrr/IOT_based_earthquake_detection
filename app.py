import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn

model = joblib.load("damage_model_tabular_hybrid_cv.pkl")
class DamageInput(BaseModel):
    lat: float
    lon: float
    epic_dist_km: float
    pga_g: float
    pgv_cms: float
    pgd_cm: float
    sa_0p3_g: float
    sa_1p0_g: float
    sa_3p0_g: float
    struct_apk_g: float


app = FastAPI(
    title="Earthquake Damage Prediction API",
    description="Backend service for predicting building damage state using a hybrid stacking model",
    version="1.0.0"
)


@app.get("/")
def home():
    return {"message": "✅ Earthquake Damage Prediction API is running!"}


@app.post("/predict")
def predict_damage(data: DamageInput):
    
    input_df = pd.DataFrame([data.dict()])

    
    pred = model.predict(input_df)[0]
    probas = model.predict_proba(input_df)[0]

    
    classes = model.named_steps["stack"].classes_
    prob_dict = {cls: float(p) for cls, p in zip(classes, probas)}

    return {
        "prediction": pred,
        "probabilities": prob_dict
    }


@app.post("/predict_batch")
def predict_damage_batch(data: List[DamageInput]):
    df = pd.DataFrame([d.dict() for d in data])
    preds = model.predict(df)
    probas = model.predict_proba(df)

    classes = model.named_steps["stack"].classes_
    results = []
    for pred, proba in zip(preds, probas):
        results.append({
            "prediction": pred,
            "probabilities": {cls: float(p) for cls, p in zip(classes, proba)}
        })
    return {"results": results}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
