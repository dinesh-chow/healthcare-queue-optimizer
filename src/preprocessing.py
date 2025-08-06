import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from transformers import AutoTokenizer
import torch
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, tokenizer_name="distilbert-base-uncased"):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            self.scaler = StandardScaler()
            self.label_encoder = LabelEncoder()
            logger.info(f"DataPreprocessor initialized with tokenizer: {tokenizer_name}")
        except Exception as e:
            logger.error(f"Error initializing DataPreprocessor: {str(e)}")
            raise
    
    def validate_dataframe(self, df, required_columns):
        """Validate that dataframe has required columns and data types"""
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for null values in critical columns
        critical_nulls = df[required_columns].isnull().sum()
        if critical_nulls.any():
            logger.warning(f"Found null values: {critical_nulls[critical_nulls > 0].to_dict()}")
        
        return True
    
    def fit_transform(self, df):
        try:
            required_cols = ['age', 'heart_rate', 'systolic_bp', 'diastolic_bp', 'history_diabetes', 'symptom_text', 'urgency_label']
            self.validate_dataframe(df, required_cols)
            
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
            logger.info(f"Successfully preprocessed {len(df)} training samples")
            return df, tokenized
        except Exception as e:
            logger.error(f"Error in fit_transform: {str(e)}")
            raise
    
    def transform(self, df):
        try:
            required_cols = ['age', 'heart_rate', 'systolic_bp', 'diastolic_bp', 'history_diabetes', 'symptom_text']
            self.validate_dataframe(df, required_cols)
            
            df = df.copy()
            # Only transform urgency_label if it exists (for training data)
            if 'urgency_label' in df.columns:
                df['urgency_encoded'] = self.label_encoder.transform(df['urgency_label'])
            
            num_cols = ['age', 'heart_rate', 'systolic_bp', 'diastolic_bp', 'history_diabetes']
            df[num_cols] = self.scaler.transform(df[num_cols])
            tokenized = self.tokenizer(
                list(df['symptom_text']),
                padding=True, truncation=True, max_length=32, return_tensors="pt"
            )
            logger.info(f"Successfully transformed {len(df)} samples")
            return df, tokenized
        except Exception as e:
            logger.error(f"Error in transform: {str(e)}")
            raise
