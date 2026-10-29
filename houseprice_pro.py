import streamlit as st
import joblib 
import pandas as pd 

# loading the model
with open('D:/CODING/ML using Python/Venv/Code/My_Model.joblib', 'rb') as f:
    model = joblib.load(f)

st.title("🏡 House Price Predictor")
st.subheader('''_This is a simple house price predictor web application for Machine learning model. which can predict the house price based on the input features._''')
with st.container(border=True):
    st.write('''**_This Machine learning model is trained on the California housing dataset.
            So this will predict the house price according to the dataset. 
            It will work for cities of foreign countries like California._**''')

def user_input():
    with st.container(border=True):
        st.write("**Please enter the following details:-**")
        income = st.number_input("What is the yearly income of families lives in the area (in thousands):- ", help='eg: 50000')
        st.session_state.houseage = st.slider("What is the age of houses in the area (in years):- ",min_value=1, max_value=60, help='eg: 15')
        averooms = st.number_input("Average number of rooms in the house (including: kitchen, bedrooms, hall, ect)", help='eg: 5')
        avebedr = st.number_input("Average number of Bedrooms in the house", help='eg: 3')
        st.session_state.population = st.number_input("What is the average population in the area: ",min_value=1, max_value=50000, help='eg: 300')
        member = st.number_input("How many peoples lives in the house: ",help='eg: 5')
        lat = st.number_input("Enter the latitude coordinate of area: ",help='Google Maps, eg: 34.5')
        long = st.number_input("Enter the longitude coordinate of area: ",help='Google Maps, eg: -118.6')

    income = income / 10000
    if long > 0:
        long = -long
    features = {
        'MedInc': [income],
        'HouseAge':[st.session_state.houseage],
        'AveRooms': [averooms],
        'AveBedrms': [avebedr],
        'Population': [st.session_state.population],
        'AveOccup': [member],
        'Latitude':[lat],
        'Longitude':[long]
    }
    
    return pd.DataFrame(features, index=[0])
    
input_df = user_input()

if 'prediction' not in st.session_state:
    st.session_state.prediction = None

# button for prediction 
#if st.button('**Prediction**', help='Click to predict the price'):
#   st.session_state.prediction = model.predict(input_df)

if st.button('**Prediction**', help='Click to predict the price'):
    if input_df.isnull().values.any() or (input_df == 0).values.any():
        st.warning(
            "Please fill all values properly before predicting the house price."
        )
    else:
        st.session_state.prediction = model.predict(input_df)

if st.session_state.prediction is not None:
    st.write(f"The Predicted price of house is:- **${st.session_state.prediction[0]*100000:.2f}**")
    # this will show the predicted price in the dollar format with 2 decimal points.

if st.button("**Clear**", help='Click to Clear the Results'):
    st.session_state.prediction = None
    st.session_state.probablity = None
    st.rerun()