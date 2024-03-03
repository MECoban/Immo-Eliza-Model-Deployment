import joblib
import pandas as pd
import json
from io import StringIO

from sklearn.preprocessing import StandardScaler


def predict(inpt):
    """
    NUMERICAL FEATURES
    total_area_sqm
    nbr_bedrooms
    primary_energy_consumption_sqm
    terrace_sqm
    surface_land_sqm
    garden_sqm
    construction_year
    nbr_frontages

    FL FEATURES
    fl_terrace
    fl_garden
    fl_furnished
    fl_open_fire
    fl_swimming_pool
    fl_double_glazing


    CATEGORICAL FEATURES
    property_type
    province
    subproperty_type
    state_building
    zip_code
    locality
    equipped_kitchen
    """

    data = pd.DataFrame.from_dict(inpt)

    model_name = "XGBRegressor"
    artifacts = joblib.load(f"api/{model_name}.joblib")

    # Unpack the artifacts
    num_features = artifacts["features"]["num_features"]
    fl_features = artifacts["features"]["fl_features"]
    cat_features = artifacts["features"]["cat_features"]
    imputer = artifacts["imputer"]
    enc = artifacts["enc"]
    # scaler = artifacts["scaler"]
    model = artifacts["model"]

    # One Hot encoding
    cat_enc = enc.transform(data[cat_features]).toarray()

    # Impute the values
    data[num_features] = imputer.transform(data[num_features])

    # Rescale the data
    # data[num_features] = scaler.transform(data[num_features])

    # data = pd.concat([pd.DataFrame(num_scaled, columns=num_features),
    data = pd.concat(
        [
            data[num_features],
            data[fl_features],
            pd.DataFrame(cat_enc, columns=enc.get_feature_names_out()),
        ],
        axis=1,
    )

    # Make predictions
    prediction_df = pd.DataFrame(model.predict(data))
    prediction = prediction_df.astype(float)

    return prediction.to_numpy()[0][0]
