import joblib
import pandas as pd
import json
from io import StringIO

def predict (num, fl, cat, inpt):

    data = pd.DataFrame.from_dict(json.loads(inpt))

    model_name = 'LinearRegression'
    artifacts = joblib.load(f"./models/{model_name}.joblib")

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

    return prediction.to_json()


if __name__ == "__main__":
    num = ["nbr_bedrooms", "total_area_sqm"]
    fl = []
    cat = []
    inpt = '{"nbr_bedrooms":[5], "total_area_sqm":[100.0]}'

    #inpt = StringIO('{"nbr_bedrooms":"5", "total_area_sqm":"100.0"}')
    #data = json.loads(inpt)
    #data=pd.DataFrame.from_dict(data)
    #print(data)
    #print(type(data))

    prediction = predict(num, fl, cat, inpt)
    print(type(prediction))
    print(prediction)
