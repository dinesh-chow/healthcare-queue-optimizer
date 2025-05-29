import streamlit as st
import pandas as pd
import torch
import matplotlib.pyplot as plt
from src.preprocessing import DataPreprocessor
from src.model import HealthcareUrgencyModel
import joblib

@st.cache_resource
def load_model_and_processor():
    model = HealthcareUrgencyModel()
    model.load_state_dict(torch.load("model/healthcare_model.pt", map_location="cpu"))
    model.eval()
    processor = joblib.load("model/data_processor.pkl")
    return model, processor

def predict_urgency(model, processor, input_df):
    df, tokenized = processor.transform(input_df)
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    tabular = torch.tensor(df[["age", "heart_rate", "systolic_bp", "diastolic_bp", "history_diabetes"]].values, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, tabular)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        preds = torch.argmax(probs, axis=1)
    df["predicted_urgency"] = processor.label_encoder.inverse_transform(preds.numpy())
    df["confidence"] = probs.max(dim=1).values.numpy()
    return df[["patient_id", "symptom_text", "predicted_urgency", "confidence"]]

def urgency_color(val):
    colors = {
        "high": "background-color: #ffcccc; color: red;",
        "medium": "background-color: #fff3cd; color: #856404;",
        "low": "background-color: #d4edda; color: green;",
    }
    return colors.get(val, "")

def main():
    st.set_page_config(page_title="Healthcare Queue Optimizer", layout="wide")
    st.title("🏥 Healthcare Emergency Queue Optimizer")
    st.markdown("Upload a CSV file with patient details to get urgency predictions and see the optimized queue.")

    st.markdown("**📋 Required Columns:** `patient_id`, `age`, `heart_rate`, `systolic_bp`, `diastolic_bp`, `history_diabetes`, `symptom_text`")

    uploaded_file = st.file_uploader("📁 Upload patient data (.csv)", type=["csv"])

    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        required_cols = {"patient_id", "age", "heart_rate", "systolic_bp", "diastolic_bp", "history_diabetes", "symptom_text"}

        if required_cols.issubset(set(input_df.columns)):
            model, processor = load_model_and_processor()
            predictions = predict_urgency(model, processor, input_df)

            # Count and chart
            urgency_counts = predictions["predicted_urgency"].value_counts()
            st.subheader("📊 Urgency Distribution")
            st.bar_chart(urgency_counts)

            # Show full predictions with colors
            st.subheader("📋 Patient Predictions")
            styled_df = predictions.style.applymap(urgency_color, subset=["predicted_urgency"])
            st.dataframe(styled_df, use_container_width=True)

            # Optimized queue
            st.subheader("🔁 Optimized Queue")
            urgency_order = {"high": 0, "medium": 1, "low": 2}
            predictions["urgency_rank"] = predictions["predicted_urgency"].map(urgency_order)
            sorted_queue = predictions.sort_values(by=["urgency_rank", "confidence"], ascending=[True, False])
            st.dataframe(sorted_queue.drop(columns=["urgency_rank"]), use_container_width=True)

        else:
            st.error("❌ CSV file is missing required columns.")
    else:
        st.info("📥 Please upload a patient CSV file to begin.")

if __name__ == "__main__":
    main()

