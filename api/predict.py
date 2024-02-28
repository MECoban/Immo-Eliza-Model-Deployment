import joblib
import pandas as pd
import json
from io import StringIO
from sklearn.preprocessing import StandardScaler

def predict (num, fl, cat, inpt):

    data = pd.DataFrame.from_dict(json.loads(inpt))

    model_name = 'XGBRegressor'
    artifacts = joblib.load(f"./models/{model_name}.joblib")

    # Unpack the artifacts
    num_features = artifacts["features"]["num_features"]
    fl_features = artifacts["features"]["fl_features"]
    cat_features = artifacts["features"]["cat_features"]
    imputer = artifacts["imputer"]
    enc = artifacts["enc"]
    scaler = artifacts["scaler"]
    model = artifacts["model"]

    # Extract the used data
    #data = data[num + fl + cat]
    #data = data.round({'latitude': 4, 'longitude': 4})

    # Rescale the data
    scaler.transform(data)

    # Make predictions
    prediction = pd.DataFrame(model.predict(data))

    return prediction.to_numpy()[0][0]


#if __name__ == "__main__":
#    num = ["nbr_bedrooms", "total_area_sqm"]
#    fl = []
#    cat = []
#    inpt = '{"nbr_bedrooms":[5], "total_area_sqm":[100.0], "surface_land_sqm":[200], "nbr_frontages":[3], "terrace_sqm":[10], "garden_sqm":[50], "construction_year":[2002]}'
#
#    #inpt = StringIO('{"nbr_bedrooms":"5", "total_area_sqm":"100.0"}')
#    #data = json.loads(inpt)
#    #data=pd.DataFrame.from_dict(data)
#    #print(data)
#    #print(type(data))
#
#    prediction = predict(num, fl, cat, inpt)
#    print(type(prediction))
#    print(prediction)
