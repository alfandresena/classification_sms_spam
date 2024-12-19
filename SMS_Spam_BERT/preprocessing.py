from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class SpamDataset(Dataset):
    """Dataset personnalisé pour les données de spam"""
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

def preparer_donnees(data, test_size=0.2, val_size=0.1, batch_size=16):
    """
    Prépare les données pour l'entraînement avec BERT.
    """
    # Nettoyage des données
    data = data[['v1', 'v2']].dropna()
    data.columns = ['label', 'message']
    
    # Encodage des labels
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})
    
    # Division des données
    X = data['message'].values
    y = data['label'].values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=True)
    
    # Train/val split
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
                                                     test_size=val_size_adjusted, 
                                                     random_state=42, stratify=True)
    
    # Création des datasets
    train_dataset = SpamDataset(X_train, y_train)
    val_dataset = SpamDataset(X_val, y_val)
    test_dataset = SpamDataset(X_test, y_test)
    
    # Création des dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Nombre d'échantillons - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader
    }
