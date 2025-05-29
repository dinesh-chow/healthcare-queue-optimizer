import pandas as pd
import numpy as np 
import random

def generate_synthetic_data(n=500):
    np.random.seed(42)
    random.seed(42)
    data = []
    for i in range(n):
        age = np.random.randint(20, 90)
        heart_rate = np.random.randint(50, 120)
        systolic_bp = np.random.randint(90, 180)
        diastolic_bp = np.random.randint(60, 120)
        history_diabetes = np.random.choice([0, 1], p=[0.85, 0.15])
        
        symptoms_pool = [
            "chest pain and dizziness", "mild headache and fatigue", "cough and sore throat",
            "shortness of breath", "abdominal pain and nausea", "fever and chills",
            "joint pain and swelling", "blurred vision", "severe back pain",
            "vomiting and diarrhea"
        ]
        symptom_text = random.choice(symptoms_pool)
        
        # Simple urgency heuristic
        if age > 65 and "chest pain" in symptom_text:
            urgency = 'high'
        elif heart_rate > 100 or systolic_bp > 140:
            urgency = 'medium'
        else:
            urgency = 'low'
        
        data.append([
            i+1, age, heart_rate, systolic_bp, diastolic_bp, history_diabetes, symptom_text, urgency
        ])
        
    df = pd.DataFrame(data, columns=[
        'patient_id', 'age', 'heart_rate', 'systolic_bp', 'diastolic_bp',
        'history_diabetes', 'symptom_text', 'urgency_label'
    ])
    return df

if __name__ == "__main__":
    df = generate_synthetic_data(500)
    df.to_csv("data/synthetic_data.csv", index=False)
    print("Synthetic dataset saved to data/synthetic_data.csv")
