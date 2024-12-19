from bert_classifier import BertClassifier
import torch
import torch.optim as optim
from sklearn.metrics import f1_score
from transformers import BertTokenizer
import torch.nn as nn

class BertModel:
    """Classe pour gérer l'entraînement et l'évaluation du modèle BERT"""
    def __init__(self, num_classes, device, dropout=0.1):
        self.device = device
        self.model = BertClassifier(num_classes=num_classes, dropout=dropout).to(device)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
    def tokenize_data(self, texts, max_length=128):
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
    
    def train(self, train_loader, val_loader, epochs=3, lr=2e-5):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            # Phase d'entraînement
            self.model.train()
            total_train_loss = 0
            train_targets = []
            train_predictions = []
            
            for batch_texts, batch_labels in train_loader:
                batch_labels = batch_labels.to(self.device)
                encoded = self.tokenize_data(batch_texts)
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
                train_targets.extend(batch_labels.cpu().numpy())
                train_predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            
            # Calcul des métriques d'entraînement
            train_f1 = f1_score(train_targets, train_predictions, average='weighted')
            
            # Phase de validation
            val_f1, val_loss = self._validate(val_loader, criterion)
            
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"  Train Loss: {total_train_loss/len(train_loader):.4f} | Train F1: {train_f1:.4f}")
            print(f"  Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")
    
    def _validate(self, val_loader, criterion):
        self.model.eval()
        total_val_loss = 0
        val_targets = []
        val_predictions = []
        
        with torch.no_grad():
            for batch_texts, batch_labels in val_loader:
                batch_labels = batch_labels.to(self.device)
                encoded = self.tokenize_data(batch_texts)
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, batch_labels)
                
                total_val_loss += loss.item()
                val_targets.extend(batch_labels.cpu().numpy())
                val_predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        
        return f1_score(val_targets, val_predictions, average='weighted'), total_val_loss/len(val_loader)
    
    def evaluate(self, test_loader):
        self.model.eval()
        test_targets = []
        test_predictions = []
        
        with torch.no_grad():
            for batch_texts, batch_labels in test_loader:
                batch_labels = batch_labels.to(self.device)
                encoded = self.tokenize_data(batch_texts)
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                test_targets.extend(batch_labels.cpu().numpy())
                test_predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        
        test_f1 = f1_score(test_targets, test_predictions, average='weighted')
        print(f"Test F1 Score: {test_f1:.4f}")
        return test_f1