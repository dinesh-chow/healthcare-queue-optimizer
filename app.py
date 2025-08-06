import streamlit as st
import pandas as pd
import torch
import matplotlib.pyplot as plt
from src.preprocessing import DataPreprocessor
from src.model import HealthcareUrgencyModel
import joblib
import os
import json
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Healthcare Queue Optimizer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

def check_model_files():
    """Check if required model files exist"""
    model_path = "model/healthcare_model.pt"
    processor_path = "model/data_processor.pkl"
    
    missing_files = []
    if not os.path.exists(model_path):
        missing_files.append(model_path)
    if not os.path.exists(processor_path):
        missing_files.append(processor_path)
    
    return missing_files

def load_training_metadata():
    """Load training metadata if available"""
    metadata_path = "model/training_metadata.json"
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load training metadata: {e}")
    return None

@st.cache_resource
def load_model_and_processor():
    """Load model and processor with error handling"""
    try:
        missing_files = check_model_files()
        if missing_files:
            st.error(f"Missing required files: {missing_files}")
            st.error("Please run the training script first: `python -m src.train`")
            st.stop()
        
        model = HealthcareUrgencyModel()
        model.load_state_dict(torch.load("model/healthcare_model.pt", map_location="cpu"))
        model.eval()
        processor = joblib.load("model/data_processor.pkl")
        
        logger.info("Model and processor loaded successfully")
        return model, processor
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        logger.error(f"Error loading model: {str(e)}")
        st.stop()

def validate_input_data(df):
    """Validate uploaded CSV data"""
    required_cols = {"patient_id", "age", "heart_rate", "systolic_bp", "diastolic_bp", "history_diabetes", "symptom_text"}
    
    # Check required columns
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        return False, f"Missing required columns: {missing_cols}"
    
    # Check data types and ranges
    issues = []
    
    # Age validation
    if not pd.api.types.is_numeric_dtype(df['age']):
        issues.append("Age must be numeric")
    elif (df['age'] < 0).any() or (df['age'] > 150).any():
        issues.append("Age must be between 0 and 150")
    
    # Heart rate validation
    if not pd.api.types.is_numeric_dtype(df['heart_rate']):
        issues.append("Heart rate must be numeric")
    elif (df['heart_rate'] < 30).any() or (df['heart_rate'] > 200).any():
        issues.append("Heart rate must be between 30 and 200")
    
    # Blood pressure validation
    for col, min_val, max_val in [('systolic_bp', 70, 250), ('diastolic_bp', 40, 150)]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            issues.append(f"{col} must be numeric")
        elif (df[col] < min_val).any() or (df[col] > max_val).any():
            issues.append(f"{col} must be between {min_val} and {max_val}")
    
    # Diabetes history validation
    if not df['history_diabetes'].isin([0, 1]).all():
        issues.append("history_diabetes must be 0 (No) or 1 (Yes)")
    
    # Symptom text validation
    if df['symptom_text'].isnull().any():
        issues.append("symptom_text cannot be empty")
    
    # Patient ID validation
    if df['patient_id'].isnull().any():
        issues.append("patient_id cannot be empty")
    
    if issues:
        return False, "Data validation errors: " + "; ".join(issues)
    
    return True, "Data validation passed"

def predict_urgency(model, processor, input_df):
    """Make urgency predictions with error handling"""
    try:
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
    except Exception as e:
        st.error(f"Error making predictions: {str(e)}")
        logger.error(f"Prediction error: {str(e)}")
        return None

def urgency_color(val):
    colors = {
        "high": "background-color: #ffcccc; color: red;",
        "medium": "background-color: #fff3cd; color: #856404;",
        "low": "background-color: #d4edda; color: green;",
    }
    return colors.get(val, "")

def main():
    # Header and description
    st.title("🏥 Healthcare Emergency Queue Optimizer")
    st.markdown("""
    **AI-powered patient triage system** that predicts urgency levels and optimizes emergency room queues in real-time.
    
    Upload patient data to get instant urgency predictions and see the optimized queue order.
    """)

    # Sidebar with model information
    with st.sidebar:
        st.header("📊 Model Information")
        metadata = load_training_metadata()
        if metadata:
            st.write(f"**Training Date:** {metadata.get('training_date', 'Unknown')[:10]}")
            st.write(f"**Model:** {metadata.get('model_architecture', 'Unknown')}")
            st.write(f"**Classes:** {', '.join(metadata.get('classes', []))}")
            st.write(f"**Training Samples:** {metadata.get('train_samples', 'Unknown')}")
            if st.button("🔄 Refresh Model Cache"):
                st.cache_resource.clear()
                st.rerun()
        else:
            st.warning("No training metadata found")

        st.header("📋 Data Requirements")
        st.markdown("""
        **Required Columns:**
        - `patient_id`: Unique identifier
        - `age`: Age in years (0-150)
        - `heart_rate`: Beats per minute (30-200)
        - `systolic_bp`: Systolic BP (70-250)
        - `diastolic_bp`: Diastolic BP (40-150)
        - `history_diabetes`: 0=No, 1=Yes
        - `symptom_text`: Description of symptoms
        
        💡 **Sample data contains 20 realistic patient records with diverse symptoms and age-appropriate vital signs.**
        """)
        
        # Sample data download
        if st.button("📥 Download Sample Data"):
            # Use actual synthetic data for sample download
            if os.path.exists("data/synthetic_data.csv"):
                sample_data = pd.read_csv("data/synthetic_data.csv")
                # Remove urgency_label column for sample data (users shouldn't see the answers)
                if 'urgency_label' in sample_data.columns:
                    sample_data = sample_data.drop(columns=['urgency_label'])
                # Take first 20 rows as sample
                sample_data = sample_data.head(20)
            else:
                # Fallback to basic sample if synthetic data not available
                sample_data = pd.DataFrame({
                    'patient_id': ['P001', 'P002', 'P003'],
                    'age': [45, 32, 67],
                    'heart_rate': [100, 85, 110],
                    'systolic_bp': [140, 120, 160],
                    'diastolic_bp': [90, 80, 95],
                    'history_diabetes': [1, 0, 1],
                    'symptom_text': [
                        'chest pain and dizziness',
                        'mild headache and nausea',
                        'severe shortness of breath'
                    ]
                })
            
            csv = sample_data.to_csv(index=False)
            st.download_button(
                label="Download sample.csv",
                data=csv,
                file_name="sample_patient_data.csv",
                mime="text/csv"
            )

    # File upload section
    st.header("📁 Upload Patient Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV file with patient data",
        type=["csv"],
        help="Upload a CSV file with the required patient information columns"
    )

    if uploaded_file is not None:
        try:
            # Load and validate data
            input_df = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded {len(input_df)} patient records")
            
            # Show data preview
            with st.expander("👀 Preview Uploaded Data", expanded=False):
                st.dataframe(input_df.head(), use_container_width=True)
                st.write(f"**Shape:** {input_df.shape[0]} rows × {input_df.shape[1]} columns")

            # Validate data
            is_valid, validation_message = validate_input_data(input_df)
            
            if not is_valid:
                st.error(f"❌ {validation_message}")
                st.stop()
            else:
                st.success("✅ Data validation passed")

            # Load model and make predictions
            with st.spinner("🤖 Loading AI model and making predictions..."):
                model, processor = load_model_and_processor()
                predictions = predict_urgency(model, processor, input_df)
            
            if predictions is not None:
                # Display results
                st.header("📊 Prediction Results")
                
                # Urgency distribution chart
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    urgency_counts = predictions["predicted_urgency"].value_counts()
                    st.subheader("🎯 Urgency Distribution")
                    
                    # Create a more detailed chart
                    fig, ax = plt.subplots(figsize=(8, 5))
                    colors = {'high': '#ff4444', 'medium': '#ffaa44', 'low': '#44ff44'}
                    bars = ax.bar(urgency_counts.index, urgency_counts.values, 
                                 color=[colors.get(x, '#cccccc') for x in urgency_counts.index])
                    ax.set_xlabel('Urgency Level')
                    ax.set_ylabel('Number of Patients')
                    ax.set_title('Patient Urgency Distribution')
                    
                    # Add value labels on bars
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{int(height)}', ha='center', va='bottom')
                    
                    st.pyplot(fig)
                
                with col2:
                    st.subheader("📈 Confidence Statistics")
                    avg_confidence = predictions["confidence"].mean()
                    min_confidence = predictions["confidence"].min()
                    max_confidence = predictions["confidence"].max()
                    
                    st.metric("Average Confidence", f"{avg_confidence:.2%}")
                    st.metric("Min Confidence", f"{min_confidence:.2%}")
                    st.metric("Max Confidence", f"{max_confidence:.2%}")
                    
                    # Confidence histogram
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.hist(predictions["confidence"], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                    ax.set_xlabel('Confidence Score')
                    ax.set_ylabel('Number of Patients')
                    ax.set_title('Confidence Score Distribution')
                    st.pyplot(fig)

                # Detailed predictions table
                st.subheader("📋 Detailed Predictions")
                styled_df = predictions.style.applymap(urgency_color, subset=["predicted_urgency"])
                st.dataframe(styled_df, use_container_width=True)

                # Optimized queue
                st.subheader("🔄 Optimized Queue Order")
                st.info("Patients are sorted by urgency level (High → Medium → Low) and then by confidence score (highest first)")
                
                urgency_order = {"high": 0, "medium": 1, "low": 2}
                predictions["urgency_rank"] = predictions["predicted_urgency"].map(urgency_order)
                sorted_queue = predictions.sort_values(by=["urgency_rank", "confidence"], ascending=[True, False])
                
                # Add queue position
                sorted_queue["queue_position"] = range(1, len(sorted_queue) + 1)
                display_queue = sorted_queue[["queue_position", "patient_id", "symptom_text", "predicted_urgency", "confidence"]]
                
                st.dataframe(display_queue, use_container_width=True)
                
                # Download results
                csv_results = sorted_queue.drop(columns=["urgency_rank"]).to_csv(index=False)
                st.download_button(
                    label="📥 Download Optimized Queue",
                    data=csv_results,
                    file_name=f"optimized_queue_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            logger.error(f"File processing error: {str(e)}")
    
    else:
        # Show instructions when no file is uploaded
        st.info("👆 Please upload a patient CSV file to begin analysis")
        
        # Show example of expected data format
        st.subheader("📝 Expected Data Format")
        
        # Use actual synthetic data for example (first 5 rows)
        if os.path.exists("data/synthetic_data.csv"):
            synthetic_df = pd.read_csv("data/synthetic_data.csv")
            # Remove urgency_label column for example (users shouldn't see the answers)
            if 'urgency_label' in synthetic_df.columns:
                synthetic_df = synthetic_df.drop(columns=['urgency_label'])
            example_df = synthetic_df.head(5)
        else:
            # Fallback to basic example if synthetic data not available
            example_df = pd.DataFrame({
                'patient_id': ['P001', 'P002', 'P003'],
                'age': [45, 32, 67],
                'heart_rate': [100, 85, 110],
                'systolic_bp': [140, 120, 160],
                'diastolic_bp': [90, 80, 95],
                'history_diabetes': [1, 0, 1],
                'symptom_text': [
                    'chest pain and dizziness',
                    'mild headache and nausea',
                    'severe shortness of breath'
                ]
            })
        
        st.dataframe(example_df, use_container_width=True)
        st.info(f"💡 This example shows {len(example_df)} patients from our realistic synthetic dataset. The actual sample download contains 20 diverse patient records.")

if __name__ == "__main__":
    main()
