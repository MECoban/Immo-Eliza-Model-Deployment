import joblib
import pandas as pd
import json
from io import StringIO

def predict (num, fl, cat, inpt):

    data = pd.DataFrame.from_dict(json.loads(inpt))

    model_name = 'LinearRegression'
    artifacts = joblib.load(f"{model_name}.joblib")

    # Unpack the artifacts
    num_features = artifacts["features"]["num_features"]
    fl_features = artifacts["features"]["fl_features"]
    cat_features = artifacts["features"]["cat_features"]
    imputer = artifacts["imputer"]
    enc = artifacts["enc"]
    model = artifacts["model"]

    # Extract the used data
    #data = data[num + fl + cat]
    #data = data.round({'latitude': 4, 'longitude': 4})

    # Make predictions
    prediction = pd.DataFrame(model.predict(data))

    return prediction.to_numpy()[0][0]