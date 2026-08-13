import streamlit as st
import joblib
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Page config
st.set_page_config(page_title="Ford Price Predictor", page_icon="🚗", layout="centered")

# CSS for styling
st.markdown("""
    <style>
   .main {background-color: #0e1117;}
   .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        border-radius: 12px;
        padding: 10px 24px;
        width: 100%;
    }
   .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
   .price-box {
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        color: #000;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🚗 Ford Car Price Predictor")
st.markdown("Choose ur ford details and get predicted price 👇")

MODEL_FILE = 'model.pkl'
DATA_FILE = 'ford.csv'

df = pd.read_csv(DATA_FILE)

# Model training
if not os.path.exists(MODEL_FILE):
    with st.spinner('Pehli baar hai... Model train ho raha hai 10 sec ruko ⏳'):
        X = df.drop('price', axis=1)
        y = df['price']
        
        cat_cols = ['model', 'transmission', 'fuelType']
        num_cols = ['year', 'mileage', 'tax', 'mpg', 'engineSize']
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
                ('num', 'passthrough', num_cols)
            ])
        
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ])
        
        model.fit(X, y)
        joblib.dump(model, MODEL_FILE)
    st.success("✅ Model ban gaya! Page refresh kar")
    st.stop()
else:
    model = joblib.load(MODEL_FILE)

# Input form in 2 columns
col1, col2 = st.columns(2)

with col1:
    model_name = st.selectbox('🚘 Model', sorted(df['model'].unique()))
    year = st.slider('📅 Year', 2000, 2025, 2019)
    km = st.number_input('📍 Mileage/KM', 0, 300000, 50000, step=1000)
    transmission = st.selectbox('⚙️ Transmission', sorted(df['transmission'].unique()))

with col2:
    fuel = st.selectbox('⛽ Fuel Type', sorted(df['fuelType'].unique()))
    tax = st.slider('💷 Tax £', 0, 1000, 150,step=10)
    mpg = st.slider('🏁 MPG', 10.0, 100.0, 50.0, step=1.0)
    engine = st.slider('🔧 Engine Size L', 0.5, 5.0, 1.0, step=0.2)

if st.button('💰 Predict Price Now'):
    input_df = pd.DataFrame([[model_name, year, km, transmission, fuel, tax, mpg, engine]], 
                            columns=['model','year','mileage','transmission','fuelType','tax','mpg','engineSize'])
    pred = model.predict(input_df)
    st.markdown(f'<div class="price-box">Predicted Price: £{pred[0]:,.2f}</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("Made with ❤️ using Streamlit + Machine Learning")