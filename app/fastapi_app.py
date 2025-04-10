from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import asyncio
import os

app = FastAPI()

# Load trained model and scaler
current_dir = os.getcwd()
model_path = os.path.join(current_dir, 'trained_models', 'Logistic Regression_model.pkl')
scaler_path = os.path.join(current_dir, 'trained_models', 'scaler.pkl')

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Define input schema
class InputData(BaseModel):
    battery_power: int
    blue: int
    clock_speed: float
    dual_sim: int
    fc: int
    four_g: int
    int_memory: int
    m_dep: float
    mobile_wt: int
    n_cores: int
    pc: int
    px_height: int
    px_width: int
    ram: int
    sc_h: int
    sc_w: int
    talk_time: int
    three_g: int
    touch_screen: int
    wifi: int

@app.post("/predict")
async def predict(data: InputData):
    # Optional: Simulate I/O delay
    await asyncio.sleep(1)

    # Convert to numpy array and reshape
    raw_features = np.array([[value for value in data.model_dump().values()]])

    # Scale input features using the same scaler used during training
    scaled_features = scaler.transform(raw_features)

    # Predict using the trained model
    prediction = model.predict(scaled_features)

    return {"prediction": int(prediction[0])}
