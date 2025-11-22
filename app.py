import streamlit as st
import json
import time

st.title("AI Circle Presentation - Demo")

st.write("This is the demo UI. The actual circle detection runs in circle_detector.py")

# Load explanations
with open("explanations.json", "r") as f:
    data = json.load(f)

term = st.selectbox("Select a term", list(data.keys()))
st.write("Explanation:")
st.info(data[term])

