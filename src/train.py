import joblib
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from yellowbrick.regressor import ResidualsPlot

def train():
    """Trains a linear regression model on the full dataset and stores output."""
    # Load the data
    data = pd.read_csv("data/properties.csv")

    # Define features to use
    num_features = ['nbr_bedrooms', 
            'total_area_sqm', 
            #'latitude', 
            #'surface_land_sqm', 
            #'nbr_frontages', 
            #'terrace_sqm', 
            #'garden_sqm',
            #'construction_year',
            #'primary_energy_consumption_sqm',
            #'cadastral_income'
            ]
    #'equiped_kitchen'

    #fl_features =['fl_swimming_pool', 
    #        'fl_furnished', 
    #        'fl_open_fire', 
    #        'fl_terrace', 
    #        'fl_garden', 
    #        'fl_swimming_pool', 
    #        'fl_floodzone',
    #        'fl_double_glazing'
    #        ]

    #cat_features = ['region', 
    #        'property_type', 
    #        'subproperty_type', 
    #        'province', 
    #        'locality', 
    #        'zip_code', 
    #        'state_building',
    #        'heating_type',
    #        ]

    # Split the data into features and target
    X = data[num_features + fl_features + cat_features]
    y = data["price"]

    # Standardize the numerical values
    scaler = StandardScaler()
    scaler.fit_tranform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=505)
    
    # Removing the outliers with the IQR method
    mult = 2   # remove a lot of stuff because the r2 will be bigger and we like that
    for X in [X_train, X_test]:
        for feat in num_features:
            Q1 = X[feat].quantile(0.15)
            Q3 = X[feat].quantile(0.85)
            IQR = Q3 - Q1
            outliers = (X[feat] < (Q1-mult*IQR)) | (X[feat] > (Q3+mult*IQR))
            X[feat] = X[feat][~outliers]

    for y in [y_train, y_test]:
        Q1 = y.quantile(0.15)
        Q3 = y.quantile(0.85)
        IQR = Q3 - Q1
        outliers = (y < (Q1-mult*IQR)) | (y > (Q3+mult*IQR))
        y = y[~outliers]

    # Impute missing values using SimpleImputer
    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_train[num_features])
    X_train[num_features] = imputer.transform(X_train[num_features])
    X_test[num_features] = imputer.transform(X_test[num_features])

    # Convert categorical columns with one-hot encoding using OneHotEncoder
    #enc = OneHotEncoder(handle_unknown='ignore')
    #enc.fit(X_train[cat_features])
    #X_train_cat = enc.transform(X_train[cat_features]).toarray()
    #X_test_cat = enc.transform(X_test[cat_features]).toarray()

    # Combine the numerical and one-hot encoded categorical columns
    #X_train = pd.concat(
    #    [
    #        X_train[num_features + fl_features].reset_index(drop=True),
    #        pd.DataFrame(X_train_cat, columns=enc.get_feature_names_out()),
    #    ],
    #    axis=1,
    #)

    #X_test = pd.concat(
    #    [
    #        X_test[num_features + fl_features].reset_index(drop=True),
    #        pd.DataFrame(X_test_cat, columns=enc.get_feature_names_out()),
    #    ],
    #    axis=1,
    #)

    print("Training the model on the train dataset...")

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    # R2 evaluation
    train_score = r2_score(y_train, model.predict(X_train))
    test_score = r2_score(y_test, model.predict(X_test))
    print(f"Train R² score: {train_score}")
    print(f"Test R² score: {test_score}")

    # Residual plot evaluation
    visualizer = ResidualsPlot(model,train_alpha=0.5, test_alpha=0.5) # Initialize the residual plot visualizer
    visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
    visualizer.score(X_test, y_test)  # Evaluate the model on the test data
    model_name = f'{model=}'.split('=')[1]
    visualizer.show(outpath=f"./eval/residual_plot/{model_name[:-2]}.png")                 # Finalize and render the figure

    # Retrain the model on the full dataset before saving it 
    print('Retraining the model on the full dataset...')
    model.fit(pd.concat([X_train, X_test]), pd.concat([y_train, y_test]))

    # Save the model
    artifacts = {
        "features": {
            "num_features": num_features,
            "fl_features": fl_features,
            "cat_features": cat_features,
        },
        "imputer": imputer,
        "enc": enc,
        "model": model,
    }

    joblib.dump(artifacts, f"../models/{model_name[:-2]}.joblib")
