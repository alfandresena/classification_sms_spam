from transformers import DebertaModel, DebertaConfig
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
import torch.optim as optim
from tqdm import tqdm

class DeBERTaClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(DeBERTaClassifier, self).__init__()
        self.deberta = DebertaModel.from_pretrained('microsoft/deberta-base')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.deberta.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[0][:, 0]  # Utiliser le token [CLS]
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

class DeBERTaModel:
    def __init__(self, num_classes, device):
        self.device = device
        self.model = DeBERTaClassifier(num_classes).to(device)
        
    def train(self, train_loader, val_loader, epochs=3, lr=2e-5):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            # Phase d'entraînement
            self.model.train()
            total_train_loss = 0
            train_predictions = []
            train_targets = []
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                
                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
                train_predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                train_targets.extend(labels.cpu().numpy())
            
            train_f1 = f1_score(train_targets, train_predictions, average='weighted')
            
            # Phase de validation
            val_loss, val_f1 = self.evaluate(val_loader, criterion)
            
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"  Train Loss: {total_train_loss/len(train_loader):.4f} | Train F1: {train_f1:.4f}")
            print(f"  Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")
    
    def evaluate(self, loader, criterion=None):
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
            
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in loader:
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                targets.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(loader)
        f1 = f1_score(targets, predictions, average='weighted')
        
        return avg_loss, f1