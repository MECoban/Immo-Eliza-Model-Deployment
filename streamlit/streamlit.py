
import streamlit as st
#from streamlit_predict import predict
#from streamlit_train import clean_data
import pandas as pd
import json 
import requests

df = pd.read_csv("api/data/properties.csv")

property_type = df["property_type"].unique().tolist()

def main():


    st.markdown("""
    <div style="text-align:center">
    <h1>Estate price predictor 📈</h1>
    </div>
    """, unsafe_allow_html=True)
    st.text("")

    st.header('Features  👀')

    FASTAPI_URL =  st.secrets["api_url"]
    #FASTAPI_URL =  "http://localhost:8000"

    total_area_sqm = st.number_input("Living Area")
    nbr_bedrooms = st.number_input('Number of bedrooms')
    primary_energy_consumption_sqm = st.number_input("Energy Consumption per sqaure meter")
    terrace_sqm = st.number_input("Terrace Area")
    surface_land_sqm = st.number_input("Surface Area")
    garden_sqm = st.number_input("Garden Area")
    construction_year = st.number_input("COnstruction Year")
    nbr_frontages = st.number_input("Number of Frontage")
    fl_terrace = st.checkbox("Terrace")
    fl_garden = st.checkbox("Garden")
    fl_furnished = st.checkbox("Furnished")
    fl_open_fire = st.checkbox("OpenFire")
    fl_swimming_pool = st.checkbox("Swimming Pool")
    fl_double_glazing = st.checkbox("Double Glazing")
    property_type = st.text_input("Property Type")
    province = st.text_input("Province")
    subproperty_type = st.text_input("Sub Property Type")
    state_building = st.text_input("Building State")
    zip_code = st.text_input("Zip Code")
    region = st.text_input("Region")
    equipped_kitchen = st.text_input("Kitchen Equipped Type")
    
    #select_prop_type = st.selectbox("Property type?", property_type)
    #terrace = st.radio('Does it have a terrace?:', [1, 0])

    #inputs = {"nbr_bedrooms": bedrooms, "total_area_sqm": living_area, "fl_terrace": terrace, "property_type": select_prop_type}
    #inputs = {"nbr_bedrooms": bedrooms, "total_area_sqm": living_area}
    #user_data = json.dumps(inputs)
    
    if st.button('Get price'):
        payload = {"total_area_sqm" : total_area_sqm,
                    "nbr_bedrooms" : nbr_bedrooms,
                    "primary_energy_consumption_sqm" : primary_energy_consumption_sqm,
                    "terrace_sqm" : terrace_sqm,
                    "surface_land_sqm" : surface_land_sqm,
                    "garden_sqm" : garden_sqm,
                    "construction_year" : construction_year,
                    "nbr_frontages" : nbr_frontages,
                    "fl_terrace" : fl_terrace,
                    "fl_garden" : fl_garden,
                    "fl_furnished" : fl_furnished,
                    "fl_open_fire" : fl_open_fire,
                    "fl_swimming_pool" : fl_swimming_pool,
                    "fl_double_glazing" : fl_double_glazing,
                    "property_type" : property_type,
                    "province" : province,
                    "subproperty_type" : subproperty_type,
                    "state_building" : state_building,
                    "zip_code" : zip_code,
                    "region" : region,
                    "equipped_kitchen" : equipped_kitchen}
        #payload = {"nbr_bedrooms": nbr_bedrooms , "total_area_sqm" : total_area_sqm   , "surface_land_sqm" : surface_land_sqm}
        response = requests.post(f"{FASTAPI_URL}/update_value", json=payload)
        if response.status_code == 200:
            #st.success("Value sent successfully!")
            result = response.json()
            # Display the result in Streamlit
            st.write("Price:", result)
        else:
            st.error("Failed to send value")

if __name__ == '__main__':
     main()

     #streamlit run streamlit.py