import joblib
import pandas as pd
import numpy as np
from scipy.stats import zscore

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

# from yellowbrick.regressor import ResidualsPlot

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV


def train():
    """Trains a linear regression model on the full dataset and stores output."""
    # Load the data
    data = pd.read_csv("api/data/properties.csv")

    # Define features to use
    num_features = [
        "total_area_sqm",
        "nbr_bedrooms",
        "primary_energy_consumption_sqm",
        "terrace_sqm",
        # "latitude", #
        # "longitude", #
        "surface_land_sqm",
        "garden_sqm",
        "construction_year",
        "nbr_frontages",
    ]

    fl_features = [
        "fl_terrace",
        "fl_garden",
        "fl_furnished",
        "fl_open_fire",
        "fl_swimming_pool",
        "fl_double_glazing",
        # "fl_floodzone"#
    ]

    cat_features = [
        "property_type",
        "province",
        "subproperty_type",
        "state_building",
        "zip_code",
        "locality",  #
        # "region",
        "equipped_kitchen",
        # "epc",
        # "heating_type",
    ]

    # Round the lat and long to 4 decimal points (increases R2)
    # data = data.round({'latitude': 4, 'longitude': 4})

    # Remove outliers for numerical data
    for column in data.select_dtypes(
        include=["float64"]
    ).columns:  # Loop through every numerical column
        z_house = np.abs(
            zscore(data.loc[data["property_type"] == "HOUSE", column])
        )  # Find zscore for houses
        z_apartment = np.abs(
            zscore(data.loc[data["property_type"] == "APARTMENT", column])
        )  # Find zscore for apartments

        # Identify outliers with a z-score greater than 3
        threshold = 3
        z = pd.concat([z_house, z_apartment]).sort_index()
        outliers = data[z > threshold]

        data = data.drop(outliers.index)  # Drop outliers by their index

    # Split the data into features and target
    X = data[num_features + fl_features + cat_features]
    y = data["price"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=505
    )

    # Impute missing values using SimpleImputer
    imputer = SimpleImputer(strategy="mean")
    imputer.fit(X_train[num_features])
    imputer.transform(X_train[num_features])
    imputer.transform(X_test[num_features])

    enc = OneHotEncoder(handle_unknown="ignore")
    # enc = LabelEncoder()
    enc.fit(X_train[cat_features])
    X_train_cat = enc.transform(X_train[cat_features]).toarray()
    X_test_cat = enc.transform(X_test[cat_features]).toarray()

    # Standardize the numerical values
    # scaler = StandardScaler()
    # scaler.fit_transform(X_train[num_features])
    # scaler.transform(X_test[num_features])

    # Convert categorical columns with one-hot encoding using OneHotEncoder
    # Combine the numerical and one-hot encoded categorical columns
    X_train = pd.concat(
        [
            X_train[num_features + fl_features].reset_index(drop=True),
            pd.DataFrame(X_train_cat, columns=enc.get_feature_names_out()),
        ],
        axis=1,
    )

    X_test = pd.concat(
        [
            X_test[num_features + fl_features].reset_index(drop=True),
            pd.DataFrame(X_test_cat, columns=enc.get_feature_names_out()),
        ],
        axis=1,
    )

    # Train the latitude longitude imputer
    # localities = [loc for loc in enc.get_feature_names_out() if 'locality' in loc]
    # localities+=['latitude', 'longitude']
    # imputer_latlong = SimpleImputer(strategy="mean")
    # imputer_latlong.fit(pd.concat([X_train[localities], X_test[localities]]))

    print("Training the model on the train dataset...")

    # Train the model

    # model = LinearRegression()
    # parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
    #          'objective':['reg:absoluteerror'],
    #          'learning_rate': [.05],#, .3, .7, 1], #so called `eta` value
    #          'max_depth': [6],#[5, 6, 7],
    #          'min_child_weight': [4],
    #          'subsample': [0.7],
    #          'colsample_bytree': [0.7],
    #          'n_estimators': [1000]}

    model_name = "XGBRegressor"
    model = XGBRegressor(
        objective="reg:absoluteerror",
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=4,
        n_estimators=1000,
    )
    model.fit(X_train, y_train)
    # model_grid = GridSearchCV(model, parameters, n_jobs = 5, verbose=True, scoring = 'r2', refit=True)
    # y_predict_train = model``.predict(X_train)
    # y_predict_test = xgb_grid.best_estimator_.predict(X_test)

    # model_grid.fit(X_train, y_train)
    # model = model_grid.best_estimator_

    # Evaluate the model
    # R2 evaluation
    train_score = r2_score(y_train, model.predict(X_train))
    print(f"Train R² score: {train_score}")
    test_score = r2_score(y_test, model.predict(X_test))
    print(f"Test R² score: {test_score}")

    train_score = mean_squared_error(y_train, model.predict(X_train))
    print(f"Train MSE score: {train_score}")
    test_score = mean_squared_error(y_train, model.predict(X_train))
    print(f"Test MSE score: {test_score}")

    ## Residual plot evaluation
    # Not available with the XGB regressor

    # #visualizer = ResidualsPlot(model,train_alpha=0.5, test_alpha=0.5) # Initialize the residual plot visualizer
    # visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
    # visualizer.score(X_test, y_test)  # Evaluate the model on the test data
    # visualizer.show(outpath=f"./eval/residual_plot/{model_name}.png")                 # Finalize and render the figure

    # Retrain the model on the full dataset before saving it
    print("Retraining the model on the full dataset...")
    model.fit(pd.concat([X_train, X_test]), pd.concat([y_train, y_test]))

    final_score = r2_score(
        pd.concat([y_train, y_test]), model.predict(pd.concat([X_train, X_test]))
    )
    print(f"Final R² score: {final_score}")
    final_score = mean_squared_error(
        pd.concat([y_train, y_test]), model.predict(pd.concat([X_train, X_test]))
    )
    print(f"Final MSE score: {final_score}")

    # Save the model
    artifacts = {
        "features": {
            "num_features": num_features,
            "fl_features": fl_features,
            "cat_features": cat_features,
        },
        "imputer": imputer,
        # "imputer_latlong": imputer_latlong,
        "enc": enc,
        # "scaler": scaler,
        "model": model,
    }

    joblib.dump(artifacts, f"api/{model_name}.joblib")


if __name__ == "__main__":
    train()
