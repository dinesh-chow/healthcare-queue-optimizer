import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from transformers import AutoTokenizer
import torch

class DataPreprocessor:
    def __init__(self, tokenizer_name="distilbert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
    
    def fit_transform(self, df):
        df = df.copy()
        # Encode labels
        df['urgency_encoded'] = self.label_encoder.fit_transform(df['urgency_label'])
        
        # Scale numerical features
        num_cols = ['age', 'heart_rate', 'systolic_bp', 'diastolic_bp', 'history_diabetes']
        df[num_cols] = self.scaler.fit_transform(df[num_cols])
        
        # Tokenize symptom text
        tokenized = self.tokenizer(
            list(df['symptom_text']),
            padding=True, truncation=True, max_length=32, return_tensors="pt"
        )
        return df, tokenized
    
    def transform(self, df):
        df = df.copy()
        df['urgency_encoded'] = self.label_encoder.transform(df['urgency_label'])
        num_cols = ['age', 'heart_rate', 'systolic_bp', 'diastolic_bp', 'history_diabetes']
        df[num_cols] = self.scaler.transform(df[num_cols])
        tokenized = self.tokenizer(
            list(df['symptom_text']),
            padding=True, truncation=True, max_length=32, return_tensors="pt"
        )
        return df, tokenized
