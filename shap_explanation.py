import shap
import torch
import numpy as np
import pandas as pd
from src.model import HealthcareUrgencyModel
from src.preprocessing import DataPreprocessor

def shap_explanation():
    # Load data and processor
    df = pd.read_csv("data/synthetic_data.csv")
    processor = DataPreprocessor()
    df, tokenized = processor.fit_transform(df)
    
    # Load model
    model = HealthcareUrgencyModel()
    model.load_state_dict(torch.load("model/healthcare_model.pt"))
    model.eval()
    
    # Select tabular data for explanation
    tabular_data = torch.tensor(df[['age', 'heart_rate', 'systolic_bp', 'diastolic_bp', 'history_diabetes']].values, dtype=torch.float32).numpy()
    
    # Create a function for SHAP to call
    def model_predict(x):
        # x shape: (batch, features)
        x_tensor = torch.tensor(x, dtype=torch.float32)
        # Fake tokenized text input (use all zero tensors)
        batch_size = x.shape[0]
        input_ids = torch.zeros((batch_size, 32), dtype=torch.long)
        attention_mask = torch.zeros((batch_size, 32), dtype=torch.long)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask, x_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1).numpy()
        return probs
    
    explainer = shap.KernelExplainer(model_predict, tabular_data[:50])
    shap_values = explainer.shap_values(tabular_data[:10])
    
    shap.summary_plot(shap_values, features=tabular_data[:10], feature_names=['age','heart_rate','systolic_bp','diastolic_bp','history_diabetes'])

if __name__ == "__main__":
    shap_explanation()
