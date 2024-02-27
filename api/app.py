from fastapi import FastAPI
#from typing import Optional
from pydantic import BaseModel
import os
from api.predict import predict
import json

PORT = os.environ.get("PORT", 8000)
app = FastAPI(port=PORT)

prediction = "Missing_value"

@app.get('/')
def index():
    return {"Status": "Alive!"}


class Data(BaseModel):
    nbr_bedrooms: float
    total_area_sqm: float



@app.post("/update_value/")
async def update_value(data: Data):
    #global prediction
    #received_value = '{{"nbr_bedrooms": "{}", "total_area_sqm": "{}"}}'.format(data.nbr_bedrooms, data.total_area_sqm)
    #prediction = predict(["nbr_bedrooms", "total_area_sqm"], [], [], received_value)
    global prediction
    prediction_dict = {"nbr_bedrooms": [data.nbr_bedrooms], "total_area_sqm": [data.total_area_sqm]}
    prediction_str = json.dumps(prediction_dict)
    prediction = predict(["nbr_bedrooms", "total_area_sqm"], [], [], prediction_str)

    return {"message": "Value received successfully", "value": prediction}


@app.get("/process_data")
def process_data():
    return prediction

#uvicorn app:app --reload