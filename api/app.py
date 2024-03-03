from fastapi import FastAPI

# from typing import Optional
from pydantic import BaseModel
import os
from api.predict import predict


PORT = os.environ.get("PORT", 8000)
# app = FastAPI(port=PORT)
app = FastAPI()

prediction = "Missing_value"


@app.get("/")
def index():
    return {"Status": "Alive!"}


class Data(BaseModel):
    total_area_sqm: float
    nbr_bedrooms: int
    primary_energy_consumption_sqm: int
    terrace_sqm: int
    surface_land_sqm: int
    garden_sqm: int
    construction_year: int
    nbr_frontages: int
    fl_terrace: int
    fl_garden: int
    fl_furnished: int
    fl_open_fire: int
    fl_swimming_pool: int
    fl_double_glazing: int
    property_type: str
    province: str
    subproperty_type: str
    state_building: str
    zip_code: str
    locality: str
    equipped_kitchen: str


@app.post("/update_value/")
async def update_value(data: Data):
    global prediction
    prediction_dict = {
        "total_area_sqm": [float(data.total_area_sqm)],
        "nbr_bedrooms": [data.nbr_bedrooms],
        "primary_energy_consumption_sqm": [data.primary_energy_consumption_sqm],
        "terrace_sqm": [data.terrace_sqm],
        "surface_land_sqm": [data.surface_land_sqm],
        "garden_sqm": [data.garden_sqm],
        "construction_year": [data.construction_year],
        "nbr_frontages": [data.nbr_frontages],
        "fl_terrace": [data.fl_terrace],
        "fl_garden": [data.fl_garden],
        "fl_furnished": [data.fl_furnished],
        "fl_open_fire": [data.fl_open_fire],
        "fl_swimming_pool": [data.fl_swimming_pool],
        "fl_double_glazing": [data.fl_double_glazing],
        "property_type": [data.property_type],
        "province": [data.province],
        "subproperty_type": [data.subproperty_type],
        "state_building": [data.state_building],
        "zip_code": [data.zip_code],
        "locality": [data.locality],
        "equipped_kitchen": [data.equipped_kitchen],
    }

    prediction = predict(prediction_dict)
    return prediction


@app.get("/process_data")
def process_data():
    return prediction


# uvicorn app:app --reload
