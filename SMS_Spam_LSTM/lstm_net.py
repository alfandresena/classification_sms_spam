import torch.nn as nn

class LSTMNet(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, num_layers, num_classes, dropout):
        super(LSTMNet, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=0)

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Fully connected output layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        x = self.embedding(x)  # Embedding
        lstm_out, _ = self.lstm(x)  # LSTM processing
        x = lstm_out[:, -1, :]  # Take the last time step
        return self.fc(x)  # Output layer