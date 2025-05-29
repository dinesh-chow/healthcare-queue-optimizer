import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report
import torch.nn.functional as F
from src.preprocessing import DataPreprocessor
from src.model import HealthcareUrgencyModel
import pandas as pd

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
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            tabular = batch['tabular'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask, tabular)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
            trues.extend(labels.cpu().numpy())
    print(classification_report(trues, preds, target_names=label_encoder.classes_))

def main():
    # Load data
    df = pd.read_csv("data/synthetic_data.csv")
    
    # Preprocess
    processor = DataPreprocessor()
    df, tokenized = processor.fit_transform(df)
    
    # Dataset & Split
    dataset = HealthcareDataset(df, tokenized)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HealthcareUrgencyModel()
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    # Training loop
    epochs = 5
    for epoch in range(epochs):
        loss = train(model, train_loader, optimizer, device)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
    
    # Evaluation
    print("Validation Results:")
    evaluate(model, val_loader, device, processor.label_encoder)
    
    # Save model & processor objects for deployment if needed
    torch.save(model.state_dict(), "model/healthcare_model.pt")
    import joblib
    joblib.dump(processor, "model/data_processor.pkl")

if __name__ == "__main__":
    main()
