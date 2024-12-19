import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

def preparer_donnees(data, max_sequence_length=100, test_size=0.2, val_size=0.1, vocab_size=5000):
    """
    Prépare les données pour l'entraînement d'un modèle PyTorch LSTM.
    
    Args:
        data (pd.DataFrame): Le dataframe contenant les données brutes.
        max_sequence_length (int): Longueur maximale des séquences de texte.
        test_size (float): Proportion des données pour l'ensemble de test.
        val_size (float): Proportion des données d'entraînement pour l'ensemble de validation.
        vocab_size (int): Taille du vocabulaire pour le tokenizer.
    
    Returns:
        dict: Contient les loaders d'entraînement, de validation et de test, ainsi que le tokenizer.
    """
    # Étape 1 : Nettoyage des données
    data = data[['v1', 'v2']].dropna()  # Conserver uniquement les colonnes importantes et supprimer les valeurs manquantes
    data.columns = ['label', 'message']  # Renommer les colonnes
    
    # Encodage des labels (ham -> 0, spam -> 1)
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})
    
    # Étape 2 : Tokenisation et padding
    tokenizer = {"<PAD>": 0, "<OOV>": 1}  # Initialiser le tokenizer avec des tokens spéciaux
    index = 2
    sequences = []
    for message in data['message']:
        sequence = []
        for word in message.split():
            if word not in tokenizer:
                if len(tokenizer) < vocab_size:  # Limiter la taille du vocabulaire
                    tokenizer[word] = index
                    index += 1
            sequence.append(tokenizer.get(word, tokenizer["<OOV>"]))
        sequences.append(sequence)
    
    # Padding des séquences
    padded_sequences = np.zeros((len(sequences), max_sequence_length), dtype=int)
    for i, seq in enumerate(sequences):
        padded_sequences[i, :min(len(seq), max_sequence_length)] = seq[:max_sequence_length]
    
    # Extraction des labels
    labels = data['label'].values
    
    # Étape 3 : Division des données en ensembles
    X_train, X_test, y_train, y_test = train_test_split(
        padded_sequences, labels, test_size=test_size, random_state=42
    )
    val_split = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_split, random_state=42
    )
    
    # Conversion en tenseurs PyTorch
    X_train_tensor = torch.tensor(X_train, dtype=torch.long)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.long)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    # Création des DataLoader
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=32, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=32, shuffle=False)
    
    print("Données préparées avec succès.")
    print(f"Ensemble d'entraînement : {X_train_tensor.shape}")
    print(f"Ensemble de validation : {X_val_tensor.shape}")
    print(f"Ensemble de test : {X_test_tensor.shape}")
    
    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "tokenizer": tokenizer
    }
