import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def predict_winner(year, category):
    """Predict winner probability from CLI"""
    try:
        # Load models
        with open('models/grammy_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)
        
        # Prepare input
        year_norm = (year - 1958) / (2024 - 1958)
        category_encoded = le.transform([category])[0]
        
        # Predict
        proba = model.predict_proba([[year_norm, category_encoded]])[0][1]
        
        print(f"\n🎵 Prediction for {category} in {year}")
        print(f"🏆 Win Probability: {proba*100:.1f}%")
        print("✅ Likely Winner" if proba > 0.5 else "❌ Likely Nominee")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("💡 Try checking available categories with: python cli.py clusters --show-all")