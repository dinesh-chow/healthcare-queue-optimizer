import pandas as pd
import numpy as np 
import random

import pandas as pd
import numpy as np 
import random
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_synthetic_data(n=500):
    """Generate diverse, realistic synthetic healthcare data"""
    np.random.seed(42)
    random.seed(42)
    
    # Expanded and more realistic symptom pools by urgency
    high_urgency_symptoms = [
        "severe chest pain and shortness of breath",
        "unconscious and unresponsive",
        "severe head trauma and bleeding",
        "severe allergic reaction and swelling",
        "heart attack symptoms with crushing chest pain",
        "stroke symptoms with facial drooping",
        "severe abdominal pain and vomiting blood",
        "difficulty breathing and turning blue",
        "severe burns over large body area",
        "compound fracture with bone exposure"
    ]
    
    medium_urgency_symptoms = [
        "moderate chest pain and dizziness",
        "high fever with severe headache",
        "persistent vomiting and dehydration",
        "severe back pain with numbness",
        "deep laceration requiring stitches",
        "possible broken bone with swelling",
        "severe allergic reaction with rash",
        "difficulty breathing during exertion",
        "severe abdominal pain and cramping",
        "persistent high blood pressure symptoms"
    ]
    
    low_urgency_symptoms = [
        "mild headache and fatigue",
        "cough and sore throat",
        "minor cut requiring bandage",
        "mild nausea and stomach upset",
        "joint pain and stiffness",
        "mild fever and cold symptoms",
        "minor rash and itching",
        "mild back pain",
        "routine medication refill",
        "follow-up appointment for healing wound"
    ]
    
    data = []
    for i in range(n):
        # More realistic age distribution (weighted toward common emergency ages)
        age_weights = np.array([0.1, 0.15, 0.2, 0.25, 0.2, 0.1])  # 20-30, 30-40, 40-50, 50-60, 60-70, 70+
        age_ranges = [(20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 90)]
        selected_range = np.random.choice(len(age_ranges), p=age_weights)
        age = np.random.randint(age_ranges[selected_range][0], age_ranges[selected_range][1])
        
        # Age-dependent vital signs with more realistic distributions
        base_hr = 70 if age < 65 else 75
        heart_rate = np.random.normal(base_hr, 15)
        heart_rate = max(50, min(150, int(heart_rate)))
        
        base_sys = 120 if age < 50 else 130 if age < 65 else 140
        base_dia = 80 if age < 50 else 85 if age < 65 else 90
        
        systolic_bp = np.random.normal(base_sys, 20)
        diastolic_bp = np.random.normal(base_dia, 10)
        systolic_bp = max(90, min(200, int(systolic_bp)))
        diastolic_bp = max(60, min(120, int(diastolic_bp)))
        
        # Age-dependent diabetes prevalence
        diabetes_prob = 0.05 if age < 30 else 0.1 if age < 50 else 0.2 if age < 65 else 0.3
        history_diabetes = np.random.choice([0, 1], p=[1-diabetes_prob, diabetes_prob])
        
        # More sophisticated urgency assignment
        urgency_weights = [0.15, 0.55, 0.30]  # high, medium, low
        urgency_choice = np.random.choice(['high', 'medium', 'low'], p=urgency_weights)
        
        if urgency_choice == 'high':
            symptom_text = random.choice(high_urgency_symptoms)
            # High urgency patients often have abnormal vitals
            if random.random() < 0.7:  # 70% chance of abnormal vitals
                heart_rate = max(heart_rate, 100) if random.random() < 0.5 else min(heart_rate, 50)
                systolic_bp = max(systolic_bp, 160) if random.random() < 0.5 else min(systolic_bp, 90)
        elif urgency_choice == 'medium':
            symptom_text = random.choice(medium_urgency_symptoms)
            # Medium urgency may have some abnormal vitals
            if random.random() < 0.4:  # 40% chance of abnormal vitals
                heart_rate = max(heart_rate, 90) if random.random() < 0.5 else heart_rate
                systolic_bp = max(systolic_bp, 140) if random.random() < 0.5 else systolic_bp
        else:  # low urgency
            symptom_text = random.choice(low_urgency_symptoms)
            # Low urgency typically has normal vitals
            heart_rate = min(heart_rate, 90)
            systolic_bp = min(systolic_bp, 140)
        
        urgency = urgency_choice
        
        data.append([
            f"P{i+1:03d}", age, heart_rate, systolic_bp, diastolic_bp, 
            history_diabetes, symptom_text, urgency
        ])
        
    df = pd.DataFrame(data, columns=[
        'patient_id', 'age', 'heart_rate', 'systolic_bp', 'diastolic_bp',
        'history_diabetes', 'symptom_text', 'urgency_label'
    ])
    
    # Log data distribution
    urgency_dist = df['urgency_label'].value_counts()
    logger.info(f"Generated {n} patients with urgency distribution: {urgency_dist.to_dict()}")
    
    return df

if __name__ == "__main__":
    logger.info("Starting synthetic data generation...")
    df = generate_synthetic_data(500)
    
    # Ensure data directory exists
    import os
    os.makedirs("data", exist_ok=True)
    
    df.to_csv("data/synthetic_data.csv", index=False)
    logger.info("Synthetic dataset saved to data/synthetic_data.csv")
    
    # Display summary statistics
    print("\n=== Dataset Summary ===")
    print(f"Total patients: {len(df)}")
    print(f"Urgency distribution:")
    print(df['urgency_label'].value_counts())
    print(f"\nAge range: {df['age'].min()}-{df['age'].max()}")
    print(f"Heart rate range: {df['heart_rate'].min()}-{df['heart_rate'].max()}")
    print(f"Diabetes prevalence: {df['history_diabetes'].mean():.2%}")
