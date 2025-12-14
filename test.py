import streamlit as st
import folium
from streamlit_folium import st_folium

st.title("Map Library Test")
st.write("This script tests if folium is installed and rendering correctly.")

try:
    # 1. Create a simple base map centered on IIT Madras
    # (Coordinates: 12.9915, 80.2336)
    m = folium.Map(location=[12.9915, 80.2336], zoom_start=15)

    # 2. Add a simple marker
    folium.Marker(
        [12.9915, 80.2336], 
        popup="IIT Madras", 
        tooltip="Test Marker"
    ).add_to(m)

    st.success("✅ Folium map object created successfully!")
    st.write("Below this line, you should see a map:")

    # 3. Render the map in Streamlit
    st_folium(m, width=700, height=500)

except Exception as e:
    st.error("❌ An error occurred:")
    st.code(str(e))