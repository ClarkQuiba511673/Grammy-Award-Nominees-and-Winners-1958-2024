import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load pre-trained model and encoders
@st.cache_data
def load_model():
    model = pickle.load(open('models/grammy_model.pkl', 'rb'))
    le = pickle.load(open('models/label_encoder.pkl', 'rb'))
    return model, le

model, le = load_model()

# Streamlit app
st.title('Grammy Award Winner Predictor')

st.write("""
This app predicts the likelihood of a Grammy nomination winning based on award category and year.
""")

# User inputs
year = st.slider('Year', 1958, 2024, 2020)
award_name = st.selectbox('Award Category', le.classes_)

# Predict button
if st.button('Predict'):
    # Prepare input
    award_encoded = le.transform([award_name])[0]
    year_norm = (year - 1958) / (2024 - 1958)
    
    # Make prediction
    prediction = model.predict_proba([[year_norm, award_encoded]])
    win_prob = prediction[0][1] * 100
    
    # Display result
    st.subheader('Prediction Result')
    st.write(f"Probability of winning: {win_prob:.1f}%")
    
    # Visualize
    if win_prob > 50:
        st.success('This nomination is predicted to WIN!')
    else:
        st.warning('This nomination is predicted to LOSE.')
    
    # Show feature importance
    st.subheader('Model Insights')
    feature_importance = pd.DataFrame({
        'Feature': ['Year', 'Award Category'],
        'Importance': model.feature_importances_
    })
    st.bar_chart(feature_importance.set_index('Feature'))