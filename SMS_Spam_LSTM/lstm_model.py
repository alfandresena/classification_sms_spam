from lstm_net import LSTMNet
from sklearn.metrics import f1_score
import torch.optim as optim
import torch
import torch.nn as nn

class LSTMModel:
    def __init__(self, input_dim, embedding_dim, hidden_dim, num_layers, num_classes, dropout, device):
        self.device = device
        self.model = LSTMNet(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout
        ).to(device)

    def train(self, train_loader, val_loader, epochs=5, lr=0.001):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            total_train_loss = 0
            train_targets = []
            train_predictions = []

            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
                train_targets.extend(batch_y.cpu().numpy())
                train_predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())

            train_f1 = f1_score(train_targets, train_predictions, average='weighted')

            # Validation phase
            self.model.eval()
            total_val_loss = 0
            val_targets = []
            val_predictions = []

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)

                    total_val_loss += loss.item()
                    val_targets.extend(batch_y.cpu().numpy())
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
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_x)
                test_targets.extend(batch_y.cpu().numpy())
                test_predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())

        test_f1 = f1_score(test_targets, test_predictions, average='weighted')
        print(f"Test F1 Score: {test_f1:.4f}")
        return test_f1