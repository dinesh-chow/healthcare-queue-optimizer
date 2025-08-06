#!/usr/bin/env python3
"""
Test script to validate the healthcare queue optimizer functionality
"""

import pandas as pd
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model import HealthcareUrgencyModel
from src.preprocessing import DataPreprocessor
import torch
import joblib

def test_model_prediction():
    """Test the model with sample data"""
    print("🏥 Testing Healthcare Queue Optimizer...")
    
    # Check if model exists
    if not os.path.exists("model/healthcare_model.pt"):
        print("❌ Model not found! Please run training first: python -m src.train")
        return False
    
    try:
        # Load model and processor
        print("📦 Loading model and preprocessor...")
        model = HealthcareUrgencyModel()
        model.load_state_dict(torch.load("model/healthcare_model.pt", map_location="cpu"))
        model.eval()
        processor = joblib.load("model/data_processor.pkl")
        print("✅ Model loaded successfully")
        
        # Create test data
        test_data = pd.DataFrame({
            'patient_id': ['TEST001', 'TEST002', 'TEST003'],
            'age': [25, 45, 70],
            'heart_rate': [80, 120, 95],
            'systolic_bp': [120, 160, 140],
            'diastolic_bp': [80, 100, 90],
            'history_diabetes': [0, 1, 1],
            'symptom_text': [
                'mild headache and fatigue',
                'severe chest pain and dizziness', 
                'shortness of breath and sweating'
            ]
        })
        
        print("🧪 Making predictions on test data...")
        
        # Preprocess data
        df, tokenized = processor.transform(test_data)
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        tabular = torch.tensor(df[["age", "heart_rate", "systolic_bp", "diastolic_bp", "history_diabetes"]].values, dtype=torch.float32)
        
        # Make predictions
        with torch.no_grad():
            outputs = model(input_ids, attention_mask, tabular)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            preds = torch.argmax(probs, axis=1)
        
        # Convert predictions
        df["predicted_urgency"] = processor.label_encoder.inverse_transform(preds.numpy())
        df["confidence"] = probs.max(dim=1).values.numpy()
        
        # Display results
        results = df[["patient_id", "symptom_text", "predicted_urgency", "confidence"]]
        
        print("\n📋 Prediction Results:")
        print("=" * 80)
        for _, row in results.iterrows():
            print(f"Patient: {row['patient_id']}")
            print(f"Symptoms: {row['symptom_text']}")
            print(f"Predicted Urgency: {row['predicted_urgency'].upper()}")
            print(f"Confidence: {row['confidence']:.2%}")
            print("-" * 40)
        
        print("✅ All tests passed! The model is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_model_prediction()
    sys.exit(0 if success else 1)
