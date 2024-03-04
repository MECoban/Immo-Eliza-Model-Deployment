import streamlit as st

# from streamlit_train import clean_data
import pandas as pd
import json
import streamlit_shadcn_ui as ui
import requests

df = pd.read_csv("api/data/properties.csv")

property_type_values = df["property_type"].unique().tolist()
province_values = df["province"].unique().tolist()
province_values.remove("MISSING")
subproperty_by_type = df.groupby("property_type")["subproperty_type"].unique()
locality_options = df.groupby("region")["locality"].unique().tolist()

belgium_localities = {
    "Antwerp": ["Antwerp", "Mechelen", "Turnhout", "Geel", "Lier", "Mortsel"],
    "East Flanders": [
        "Gent",
        "Aalst",
        "Sint-Niklaas",
        "Dendermonde",
        "Oudenaarde",
        "Ronse",
    ],
    "Flemish Brabant": ["Leuven", "Vilvoorde", "Tienen", "Diest", "Aarschot", "Halle"],
    "Limburg": ["Hasselt", "Genk", "Tongeren", "Sint-Truiden", "Lommel", "Bilzen"],
    "West Flanders": ["Brugge", "Kortrijk", "Ostend", "Roeselare", "Waregem", "Ieper"],
    "Hainaut": ["Mons", "Charleroi", "Tournai", "Mouscron", "La Louvière", "Thuin"],
    "Liège": ["Liège", "Verviers", "Huy", "Seraing", "Herstal", "Waremme"],
    "Luxembourg": [
        "Arlon",
        "Marche-en-Famenne",
        "Bastogne",
        "Durbuy",
        "Vielsalm",
        "La Roche-en-Ardenne",
    ],
    "Namur": [
        "Namur",
        "Dinant",
        "Philippeville",
        "Rochefort",
        "Ciney",
        "Fosses-la-Ville",
    ],
    "Walloon Brabant": [
        "Wavre",
        "Nivelles",
        "Tubize",
        "Jodoigne",
        "Genappe",
        "Ottignies-Louvain-la-Neuve",
    ],
    "Brussels": [
        "Anderlecht",
        "Brussels",
        "Elsene",
        "Etterbeek",
        "Evere",
        "Ganshoren",
        "Jette",
        "Koekelberg",
        "Oudergem",
        "Schaerbeek",
        "Sint-Agatha-Berchem",
        "Sint-Gillis",
        "Sint-Jans-Molenbeek",
        "Sint-Joost-ten-Node",
        "Sint-Lambrechts-Woluwe",
        "Sint-Pieters-Woluwe",
        "Uccle",
        "Vorst",
        "Watermaal-Bosvoorde",
    ],
}

property_type_mapping = {
    "APARTMENT": {
        "Apartment": "APARTMENT",
        "Duplex": "DUPLEX",
        "Triplex": "TRIPLEX",
        "Penthouse": "PENTHOUSE",
        "Loft": "LOFT",
        "Studio": "FLAT_STUDIO",
        "Ground Floor": "GROUND_FLOOR",
    },
    "HOUSE": {
        "House": "HOUSE",
        "Villa": "VILLA",
        "Manor House": "MANOR_HOUSE",
        "Country Cottage": "COUNTRY_COTTAGE",
        "Town House": "TOWN_HOUSE",
        "Mansion": "MANSION",
        "Farmhouse": "FARMHOUSE",
        "Bungalow": "BUNGALOW",
    },
}


mapping_simplified = {
    "Hyper Equipped": "HYPER_EQUIPPED",
    "Installed": "INSTALLED",
    "Semi Equipped": "SEMI_EQUIPPED",
    "Not Installed": "NOT_INSTALLED",
}

building_state_mapping = {
    "Excellent Condition": "AS_NEW",
    "Good": "GOOD",
    "To Renovate": "TO_BE_DONE_UP",
    "Just Renovated": "JUST_RENOVATED",
    "To Restore": "TO_RESTORE",
}


def main():
    st.markdown(
        """
     <style>
     input[type="checkbox"]:checked {
     color: green !important;
     }
     </style>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
     <div style="text-align:center">
     <h1>Fivers Price Predictor 🏡</h1>
     </div>
     """,
        unsafe_allow_html=True,
    )
    st.text("")
    st.divider()
    st.header("Property specifications")

    # FASTAPI_URL = "http://localhost:8000"
    # FASTAPI_URL =  "https://immo-eliza-api-zq5r.onrender.com"
    FASTAPI_URL = st.secrets["api_url"]

    property_type, blank3, blank4, blank5 = st.columns(4)
    with property_type:
        property_type = st.radio(
            label="**Select Property Type**",
            options=["Apartment", "House"],
            horizontal=True,
            index=None,
        )
    with blank3:
        st.write("")
    with blank4:
        st.write("")
    with blank5:
        st.write("")

    if property_type == "House":
        subproperty_type = st.selectbox(
            "**Subproperty Type**",
            options=property_type_mapping["HOUSE"],
            index=None,
            placeholder="Select Subproperty...",
        )
    else:
        subproperty_type = st.selectbox(
            "**Subproperty Type**",
            options=property_type_mapping["APARTMENT"],
            index=None,
            placeholder="Select Subproperty...",
        )

    province, zip_code, locality = st.columns(3)
    with zip_code:
        zip_code = st.text_input(
            "**Postal Code**", placeholder="Ex. 1000", max_chars=4, key=int
        )
    with province:
        province = st.selectbox(
            "**Province**",
            options=province_values,
            index=None,
            placeholder="Ex. Brussels",
        )
    with locality:
        if province != None:
            locality = st.selectbox(
                "**Locality**",
                options=belgium_localities[province],
                index=None,
                placeholder="Choose a locality",
            )
        else:
            locality = st.selectbox(
                "**Locality**",
                options=belgium_localities,
                index=None,
                placeholder="Ex. Uccle",
            )

    total_area_sqm, surface_land_sqm = st.columns(2)
    with total_area_sqm:
        total_area_sqm = st.text_input("**Living Area (m²)**")
    with surface_land_sqm:
        if property_type == "House":
            surface_land_sqm = surface_land_sqm = st.text_input("Total Land Area (m²)")
        else:
            surface_land_sqm = total_area_sqm

    fl_terrace, fl_garden, blank1, blank2 = st.columns(4)
    with fl_terrace:
        fl_terrace = st.checkbox("**Terrace**")
        if fl_terrace:
            terrace_sqm = st.text_input("**Terrace Area (m²)**")
        else:
            terrace_sqm = 0

    with fl_garden:
        fl_garden = st.checkbox("**Garden**")
        if fl_garden:
            garden_sqm = st.text_input("**Garden Area (m²)**")
        else:
            garden_sqm = 0
    with blank1:
        st.write("")
    with blank2:
        st.write("")

    nbr_bedrooms, equipped_kitchen = st.columns(2)
    with nbr_bedrooms:
        nbr_bedrooms = st.number_input("**Number of Bedrooms**", min_value=0, step=1)
    with equipped_kitchen:
        equipped_kitchen = st.selectbox(
            "**Kitchen Type**",
            options=mapping_simplified.keys(),
            index=None,
            placeholder="Select Kitchen Type...",
        )

    state_building, construction_year, primary_energy_consumption_sqm = st.columns(3)
    with state_building:
        state_building = st.selectbox(
            "**Building State**",
            options=building_state_mapping.keys(),
            index=None,
            placeholder="Select Building State...",
        )
    with construction_year:
        construction_year = st.text_input(
            "**Construction Year**", placeholder="Ex. 1990", max_chars=4
        )
    with primary_energy_consumption_sqm:
        primary_energy_consumption_sqm = st.text_input(
            "**Energy Consumption (kWh/m²)**"
        )

    st.divider()

    st.write("**Other Amenities Influencing the Price**")
    fl_swimming_pool, fl_double_glazing, fl_open_fire, fl_furnished = st.columns(4)

    with fl_swimming_pool:
        fl_swimming_pool = st.checkbox("**Swimming Pool**")
        if not fl_swimming_pool:
            fl_swimming_pool = 0
    with fl_double_glazing:
        fl_double_glazing = st.checkbox("**Double Glazing**")
        if not fl_double_glazing:
            fl_double_glazing = 0
    with fl_open_fire:
        fl_open_fire = st.checkbox("**Open Fire**")
        if not fl_open_fire:
            fl_open_fire = 0

    with fl_furnished:
        fl_furnished = st.checkbox("**Furnished**")
        if not fl_furnished:
            fl_furnished = 0

    nbr_frontages = st.number_input("**Number of Frontages**", 0, 4, 1)

    if st.button("Calculate Price"):
        payload = {
            "total_area_sqm": total_area_sqm,
            "nbr_bedrooms": nbr_bedrooms,
            "primary_energy_consumption_sqm": primary_energy_consumption_sqm,
            "terrace_sqm": terrace_sqm,
            "surface_land_sqm": surface_land_sqm,
            "garden_sqm": garden_sqm,
            "construction_year": construction_year,
            "nbr_frontages": nbr_frontages,
            "fl_terrace": fl_terrace,
            "fl_garden": fl_garden,
            "fl_furnished": fl_furnished,
            "fl_open_fire": fl_open_fire,
            "fl_swimming_pool": fl_swimming_pool,
            "fl_double_glazing": fl_double_glazing,
            "property_type": property_type,
            "province": province,
            "subproperty_type": subproperty_type,
            "state_building": (
                building_state_mapping[state_building]
                if state_building != None
                else None
            ),
            "zip_code": zip_code,
            "locality": locality,
            "equipped_kitchen": (
                mapping_simplified[equipped_kitchen]
                if equipped_kitchen != None
                else None
            ),
        }
        print(payload)
        response = requests.post(f"{FASTAPI_URL}/update_value", json=payload)
        if response.status_code == 200:
            # st.success("Value sent successfully!")
            result = response.json()
            formatted_number = f" {result:,.0f}"
            # Display the result in Streamlit
            st.write(f"### **Estimated Price: {formatted_number}€**")
        else:
            st.error("Failed to send value")


if __name__ == "__main__":
    main()
