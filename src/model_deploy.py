from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="API Modelo Créditos")

# ========================
# Cargar modelo
# ========================

root = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(root, "best_model.joblib")

model = joblib.load(model_path)

# ========================
# Esquema entrada
# ========================

class InputData(BaseModel):
    data: list

# ========================
# Endpoint
# ========================

@app.post("/predict")
@app.post("/predict")
def predict(input_data: InputData):

    df = pd.DataFrame(input_data.data)

    # columnas usadas en entrenamiento
    expected_cols = model.feature_names_in_

    # agregar columnas faltantes
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0

    # ordenar columnas
    df = df[expected_cols]

    preds = model.predict(df)

    return {"predictions": preds.tolist()}