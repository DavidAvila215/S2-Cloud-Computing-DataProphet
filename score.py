import numpy
import json
import joblib
import pandas as pd
from azureml.core.model import Model

def init():
    global model
    # Retrieve the model from the workspace using the registered model name
    model_path = Model.get_model_path('bankrupt_predict')
    model = joblib.load(model_path)

def run(raw_data):
    try:
        # Parse the incoming JSON data to a DataFrame.
        data = json.loads(raw_data)['data'][0]
        data = pd.DataFrame(data)

        # Make predictions using the loaded model
        result = model.predict(data).tolist()
        return json.dumps(result)
    except Exception as e:
        return json.dumps(str(e))
