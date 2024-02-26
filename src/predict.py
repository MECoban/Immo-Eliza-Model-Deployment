import joblib
import pandas as pd

class Predictor:


    def __init__(self, num_predict, fl_predict, cat_predict, data, prediction):

        self.num_predict = num_predict
        self.fl_predict = fl_predict
        self.cat_predict = cat_predict

        self.data = pd.DataFrame()
        self.prediction = pd.DataFrame()


    def take (self, path):

        self.data = pd.read_json(path)


    def give (self, prediciton):

        return prediction.to_json()


    def predict (self, data, prediction):

        model_name = 'LinearRegression'
    
        artifacts = joblib.load(f"../models/{model_name}.joblib")
    
        # Unpack the artifacts
        num_features = artifacts["features"]["num_features"]
        fl_features = artifacts["features"]["fl_features"]
        cat_features = artifacts["features"]["cat_features"]
        imputer = artifacts["imputer"]
        enc = artifacts["enc"]
        model = artifacts["model"]
    
        # Extract the used data
        data = data[num_features + fl_features + cat_features]
    
    
        # Make predictions
        prediction = model.predict(data[num_predict + fl_predict + cat_predict])
