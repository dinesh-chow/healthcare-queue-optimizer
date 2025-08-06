import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
import torch.nn.functional as F
from src.preprocessing import DataPreprocessor
from src.model import HealthcareUrgencyModel
import pandas as pd
import logging
import os
import json
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthcareDataset(Dataset):
    def __init__(self, df, tokenized):
        self.df = df.reset_index(drop=True)
        self.input_ids = tokenized['input_ids']
        self.attention_mask = tokenized['attention_mask']
        self.tabular = torch.tensor(df[['age', 'heart_rate', 'systolic_bp', 'diastolic_bp', 'history_diabetes']].values, dtype=torch.float32)
        self.labels = torch.tensor(df['urgency_encoded'].values, dtype=torch.long)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'tabular': self.tabular[idx],
            'labels': self.labels[idx]
        }

def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        tabular = batch['tabular'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask, tabular)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device, label_encoder):
    model.eval()
    preds = []
    trues = []
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            tabular = batch['tabular'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask, tabular)
            loss = F.cross_entropy(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
            trues.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    
    # Detailed evaluation metrics
    print(f"Validation Loss: {avg_loss:.4f}")
    print("\nClassification Report:")
    print(classification_report(trues, preds, target_names=label_encoder.classes_))
    print("\nConfusion Matrix:")
    print(confusion_matrix(trues, preds))
    
    return avg_loss, preds, trues

def main():
    try:
        logger.info("Starting model training pipeline...")
        
        # Check if data file exists
        data_path = "data/synthetic_data.csv"
        if not os.path.exists(data_path):
            logger.error(f"Data file not found: {data_path}")
            raise FileNotFoundError(f"Please run synthetic_data.py first to generate training data")
        
        # Load data
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} samples")
        
        # Preprocess
        logger.info("Initializing data preprocessor...")
        processor = DataPreprocessor()
        df, tokenized = processor.fit_transform(df)
        
        # Dataset & Split
        logger.info("Creating datasets and data loaders...")
        dataset = HealthcareDataset(df, tokenized)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
        logger.info(f"Training samples: {train_size}, Validation samples: {val_size}")
        
        # Model setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        model = HealthcareUrgencyModel()
        model.to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        
        # Training loop
        epochs = 5
        train_losses = []
        val_losses = []
        
        logger.info(f"Starting training for {epochs} epochs...")
        for epoch in range(epochs):
            train_loss = train(model, train_loader, optimizer, device)
            train_losses.append(train_loss)
            logger.info(f"Epoch {epoch+1}/{epochs}, Training Loss: {train_loss:.4f}")
            
            # Validate every epoch
            val_loss, _, _ = evaluate(model, val_loader, device, processor.label_encoder)
            val_losses.append(val_loss)
        
        # Final evaluation
        logger.info("Final validation results:")
        val_loss, preds, trues = evaluate(model, val_loader, device, processor.label_encoder)
        
        # Create model directory if it doesn't exist
        os.makedirs("model", exist_ok=True)
        
        # Save model and processor
        model_path = "model/healthcare_model.pt"
        processor_path = "model/data_processor.pkl"
        
        torch.save(model.state_dict(), model_path)
        import joblib
        joblib.dump(processor, processor_path)
        
        # Save training metadata
        metadata = {
            "training_date": datetime.now().isoformat(),
            "model_architecture": "HealthcareUrgencyModel",
            "tokenizer": "distilbert-base-uncased",
            "epochs": epochs,
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1],
            "train_samples": train_size,
            "val_samples": val_size,
            "device": str(device),
            "classes": processor.label_encoder.classes_.tolist()
        }
        
        with open("model/training_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Preprocessor saved to {processor_path}")
        logger.info("Training metadata saved to model/training_metadata.json")
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
