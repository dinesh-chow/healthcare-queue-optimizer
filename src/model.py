import torch
import torch.nn as nn
from transformers import AutoModel

class HealthcareUrgencyModel(nn.Module):
    def __init__(self, transformer_model="distilbert-base-uncased", tabular_features=5, hidden_dim=64, num_classes=3):
        super(HealthcareUrgencyModel, self).__init__()
        self.text_model = AutoModel.from_pretrained(transformer_model)
        self.tabular_layer = nn.Sequential(
            nn.Linear(tabular_features, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + self.text_model.config.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, input_ids, attention_mask, tabular_data):
        text_output = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_embeds = text_output.last_hidden_state[:, 0, :]  # CLS token
        tab_embeds = self.tabular_layer(tabular_data)
        combined = torch.cat([text_embeds, tab_embeds], dim=1)
        output = self.classifier(combined)
        return output
