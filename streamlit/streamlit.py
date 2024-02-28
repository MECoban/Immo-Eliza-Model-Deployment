import streamlit as st
#from streamlit_predict import predict
#from streamlit_train import clean_data
import pandas as pd
import json 
import requests

#df = pd.read_csv("../data/properties.csv")

#property_type = df["property_type"].unique().tolist()

def main():


     st.markdown("""
     <div style="text-align:center">
     <h1>Estate price predictor 📈</h1>
     </div>
     """, unsafe_allow_html=True)
     st.text("")

     st.header('Features  👀')

     FASTAPI_URL = "https://immo-eliza-api-zq5r.onrender.com"

     nbr_bedrooms = st.slider('Number of bedrooms', 0, 35, 1)
     total_area_sqm = st.slider('What\'s the living area in m²', 0, 500, 1)
     #select_prop_type = st.selectbox("Property type?", property_type)
     #terrace = st.radio('Does it have a terrace?:', [1, 0])

     #inputs = {"nbr_bedrooms": bedrooms, "total_area_sqm": living_area, "fl_terrace": terrace, "property_type": select_prop_type}
     #inputs = {"nbr_bedrooms": bedrooms, "total_area_sqm": living_area}
     #user_data = json.dumps(inputs)

     if st.button('Get price'):
        payload = {"nbr_bedrooms": nbr_bedrooms , "total_area_sqm" : total_area_sqm}
        response = requests.post(f"{FASTAPI_URL}/update_value", json=payload)
        if response.status_code == 200:
            #st.success("Value sent successfully!")
            response = requests.get(f"{FASTAPI_URL}/process_data")

            # Check if the request was successful
            if response.status_code == 200:
                result = response.json()
                # Display the result in Streamlit
                st.write("Price:", result)
            else:
                st.error(f"Error: {response.status_code}")
        else:
            st.error("Failed to send value")

if __name__ == '__main__':
     main()

     #streamlit run streamlit.py