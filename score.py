
import json
import joblib
import numpy as np
import pandas as pd
from azureml.core.model import Model

def init():
    global model
    model_path = Model.get_model_path("bankrupt_predict")
    print(f"🔁 Loading model from: {model_path}")
    model = joblib.load(model_path)


def run(raw_data):
    try:
        data = json.loads(raw_data)["data"]
        df = pd.DataFrame(data)
        prediction = model.predict(df).tolist()
        return json.dumps(prediction)
    except Exception as e:
        return json.dumps({"error": str(e)})
