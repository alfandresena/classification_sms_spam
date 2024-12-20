import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaConfig
from sklearn.metrics import f1_score
import torch.optim as optim

class RoBERTaClassifier(nn.Module):
    def __init__(self, num_classes):
        super(RoBERTaClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.roberta.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        return self.fc(x)

class RoBERTaModel:
    def __init__(self, num_classes, device):
        self.device = device
        self.model = RoBERTaClassifier(num_classes).to(device)
        
    def train(self, train_loader, val_loader, epochs=3, lr=2e-5):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            # Phase d'entraînement
            self.model.train()
            total_train_loss = 0
            train_targets = []
            train_predictions = []
            
            for batch_input_ids, batch_attention_mask, batch_labels in train_loader:
                batch_input_ids = batch_input_ids.to(self.device)
                batch_attention_mask = batch_attention_mask.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_input_ids, batch_attention_mask)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
                train_targets.extend(batch_labels.cpu().numpy())
                train_predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            
            train_f1 = f1_score(train_targets, train_predictions, average='weighted')
            
            # Phase de validation
            self.model.eval()
            total_val_loss = 0
            val_targets = []
            val_predictions = []
            
            with torch.no_grad():
                for batch_input_ids, batch_attention_mask, batch_labels in val_loader:
                    batch_input_ids = batch_input_ids.to(self.device)
                    batch_attention_mask = batch_attention_mask.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    outputs = self.model(batch_input_ids, batch_attention_mask)
                    loss = criterion(outputs, batch_labels)
                    
                    total_val_loss += loss.item()
                    val_targets.extend(batch_labels.cpu().numpy())
                    val_predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            
            val_f1 = f1_score(val_targets, val_predictions, average='weighted')
            
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"  Train Loss: {total_train_loss / len(train_loader):.4f} | Train F1 Score: {train_f1:.4f}")
            print(f"  Val Loss: {total_val_loss / len(val_loader):.4f} | Val F1 Score: {val_f1:.4f}")
    
    def evaluate(self, test_loader):
        self.model.eval()
        test_targets = []
        test_predictions = []
        
        with torch.no_grad():
            for batch_input_ids, batch_attention_mask, batch_labels in test_loader:
                batch_input_ids = batch_input_ids.to(self.device)
                batch_attention_mask = batch_attention_mask.to(self.device)
                
                outputs = self.model(batch_input_ids, batch_attention_mask)
                test_targets.extend(batch_labels.cpu().numpy())
                test_predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        
        test_f1 = f1_score(test_targets, test_predictions, average='weighted')
        print(f"Test F1 Score: {test_f1:.4f}")
        return test_f1