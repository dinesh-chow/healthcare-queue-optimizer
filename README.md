# 🏥 Healthcare Emergency Queue Optimizer

An AI-powered system that predicts the urgency level of incoming patients based on their symptoms and vital signs — enabling hospitals to optimize their consultation queue in real time.

---

## 🚀 Features

- ✅ **Multimodal ML**: Combines vital signs + symptom text using BERT + MLP
- ✅ **Real-Time Urgency Prediction**: High, Medium, Low with confidence scores
- ✅ **Smart Queue Optimization**: Automatically prioritizes based on urgency & confidence
- ✅ **Interactive Web UI**: Professional Streamlit interface for healthcare workers
- ✅ **Data Validation**: Comprehensive input validation and error handling
- ✅ **Model Interpretability**: SHAP explanations for predictions
- ✅ **Robust Training Pipeline**: Error handling, logging, and model versioning
- ✅ **Diverse Synthetic Data**: Realistic patient data with proper distributions

---

## 🧪 Model Inputs

- **Age**: Patient age (0-150 years)
- **Heart Rate**: Beats per minute (30-200 bpm)
- **Blood Pressure**: Systolic (70-250) / Diastolic (40-150) mmHg
- **Diabetes History**: 0 = No, 1 = Yes
- **Symptom Description**: Free-text description of patient symptoms

---

## 📁 Project Structure

```
healthcare-queue-optimizer/
├── app.py                     # Enhanced Streamlit web application
├── config.py                  # Configuration management
├── synthetic_data.py          # Improved data generation with realism
├── shap_explanation.py        # Model interpretability
├── model/                     # Trained model artifacts
│   ├── healthcare_model.pt    # PyTorch model weights
│   ├── data_processor.pkl     # Preprocessing pipeline
│   └── training_metadata.json # Training information & metrics
├── src/                       # Core ML pipeline
│   ├── preprocessing.py       # Enhanced data processing with validation
│   ├── model.py              # PyTorch model architecture
│   └── train.py              # Robust training with logging & error handling
├── data/                     
│   └── synthetic_data.csv    # Generated training dataset
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

---

## 🛠️ Setup Instructions

### 🚀 **Option 1: One-Click Setup (Recommended)**

**For Windows users:**
```bash
start.bat
```

**For Linux/Mac users:**
```bash
./start.sh
```

**For Python users:**
```bash
python run.py
```

This will automatically:
- ✅ Install all dependencies
- ✅ Generate training data (if needed)
- ✅ Train the AI model (if needed)
- ✅ Launch the web application

### ⚙️ **Option 2: Manual Setup**

### ✅ 1. Clone the Project
```bash
git clone https://github.com/your-username/healthcare-queue-optimizer.git
cd healthcare-queue-optimizer
```

### ✅ 2. Setup Python Virtual Environment
```bash
python -m venv .venv
.venv\Scripts\activate   # (on Windows)
# source .venv/bin/activate  # (on macOS/Linux)
```

### ✅ 3. Install Requirements
```bash
pip install -r requirements.txt
```

### ✅ 4. Generate Training Data
```bash
python synthetic_data.py
```

### ✅ 5. Train the Model
```bash
python -m src.train
```

### ✅ 6. Launch Web Application
```bash
streamlit run app.py
```

---

## 💡 Key Improvements Made

### 🔧 **Technical Enhancements**
- **Error Handling**: Comprehensive try-catch blocks throughout
- **Data Validation**: Input validation with meaningful error messages
- **Logging**: Structured logging for debugging and monitoring
- **Configuration Management**: Centralized config file for all settings
- **Model Versioning**: Training metadata and model artifacts tracking

### 📊 **Data Quality**
- **Realistic Synthetic Data**: Age-dependent vitals and symptoms
- **Diverse Symptom Pool**: Expanded to 30+ realistic symptoms by urgency
- **Proper Distributions**: Statistical realism in patient demographics
- **Smart Labeling**: Sophisticated urgency assignment logic

### 🎨 **User Experience**
- **Professional UI**: Clean, medical-focused interface design
- **Data Preview**: Upload validation and data preview functionality
- **Visual Analytics**: Charts, confidence metrics, and distribution plots
- **Download Features**: Export optimized queues and sample data
- **Real-time Feedback**: Progress indicators and status messages

### 🏗️ **Code Quality**
- **Modular Design**: Clean separation of concerns
- **Type Safety**: Better data type validation
- **Documentation**: Comprehensive inline documentation
- **Git Management**: Proper .gitignore and repository structure

---

## 📊 Sample CSV Format

```csv
patient_id,age,heart_rate,systolic_bp,diastolic_bp,history_diabetes,symptom_text
P001,45,100,140,90,1,"severe chest pain and shortness of breath"
P002,32,85,120,80,0,"mild headache and nausea"
P003,67,110,160,95,1,"difficulty breathing during exertion"
```

---

## 🔬 Model Performance

- **Architecture**: DistilBERT + MLP hybrid model
- **Training Accuracy**: ~100% on validation set
- **Confidence Scoring**: Softmax probabilities for prediction certainty
- **Classes**: High, Medium, Low urgency levels

---

## 🌐 Live Demo

**Streamlit Cloud**: [Healthcare Queue Optimizer](https://dinesh-chow-q-optimizer-2285.streamlit.app/)

---

## 🚀 Usage Examples

### **Command Line Training**
```bash
# Generate diverse training data
python synthetic_data.py

# Train with enhanced pipeline
python -m src.train

# Run SHAP explanations
python shap_explanation.py
```

### **Web Interface**
1. Upload patient CSV file
2. View real-time urgency predictions
3. Analyze confidence distributions
4. Download optimized queue order

---

## 🔮 Future Enhancements

- [ ] **Database Integration**: PostgreSQL/MongoDB for patient records
- [ ] **Real-time API**: REST API for hospital system integration
- [ ] **Advanced ML**: Ensemble models and deep learning improvements
- [ ] **Mobile App**: React Native companion app
- [ ] **Analytics Dashboard**: Historical trends and hospital metrics
- [ ] **Multi-language Support**: Internationalization features

---

## 📝 License

MIT License

---

**Built with 💙 by dinesh-chow**

*Empowering healthcare professionals with AI-driven patient triage solutions.*
