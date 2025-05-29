# 🏥 Healthcare Emergency Queue Optimizer

An AI-powered system that predicts the urgency level of incoming patients based on their symptoms and vital signs — enabling hospitals to optimize their consultation queue in real time.

---

## 🚀 Features

- ✅ Multimodal ML: Combines vital signs + symptom text
- ✅ Real-Time Urgency Prediction: High, Medium, Low
- ✅ Confidence Scores
- ✅ Queue Sorting: Automatically prioritizes based on urgency
- ✅ Interactive UI with Streamlit
- ✅ Clean, demo-ready frontend (doctor-facing)
- ✅ Custom deep learning model using BERT + MLP

---

## 🧪 Model Inputs

- Age
- Heart rate
- Blood pressure (systolic/diastolic)
- Diabetes history (0 = No, 1 = Yes)
- Free-text symptom description
  
---


## 📁 Project Structure
healthcare-queue-optimizer/
├── app.py # Streamlit app
├── synthetic_data.py # Generates training data
├── model/
│ ├── healthcare_model.pt # Trained model
│ └── data_processor.pkl # Preprocessing pipeline
├── src/
│ ├── preprocessing.py # Data normalization + tokenizer
│ ├── model.py # PyTorch model architecture
│ └── train.py # Model training script
├── data/
│ └── synthetic_data.csv # Sample dataset
├── requirements.txt # Python dependencies
└── README.md # You're reading it

---


---

## 🛠️ Setup Instructions

### ✅ 1. Clone the Project
   git clone https://github.com/your-username/healthcare-queue-optimizer.git
   cd healthcare-queue-optimizer
   
### ✅ 2. Setup Python Virtual Environment
   python -m venv venv
   venv\Scripts\activate   # (on Windows)
   
### ✅ 3. Install Requirements
   pip install -r requirements.txt
   
### ✅ 4. Generate Data & Train Model
   python synthetic_data.py
   python -m src.train
   
### ✅ 5. Run Streamlit App   
   streamlit run app.py
   
** 🧠 Sample CSV Format **
    patient_id,age,heart_rate,systolic_bp,diastolic_bp,history_diabetes,symptom_text
    P001,45,100,140,90,1,"chest pain and dizziness"
    P002,32,85,120,80,0,"mild headache and nausea"


#** WEBSITE LINK : https://dinesh-chow-q-optimizer-2285.streamlit.app/

MIT License

Built with 💙 by dinesh-chow
