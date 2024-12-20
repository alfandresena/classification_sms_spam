import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from transformers import RobertaTokenizer

def preparer_donnees(data, max_length=128, test_size=0.2, val_size=0.1, batch_size=32):
    """
    Prépare les données pour l'entraînement d'un modèle RoBERTa.
    
    Args:
        data (pd.DataFrame): Le dataframe contenant les données brutes.
        max_length (int): Longueur maximale des séquences de texte.
        test_size (float): Proportion des données pour l'ensemble de test.
        val_size (float): Proportion des données d'entraînement pour l'ensemble de validation.
        batch_size (int): Taille des lots pour les DataLoader.
    
    Returns:
        dict: Contient les loaders d'entraînement, de validation et de test, ainsi que le tokenizer.
    """
    # Nettoyage des données
    data = data[['v1', 'v2']].dropna()
    data.columns = ['label', 'message']
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})
    
    # Initialisation du tokenizer RoBERTa
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    # Tokenisation des messages
    encodings = tokenizer(
        data['message'].tolist(),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    # Création des tenseurs
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']
    labels = torch.tensor(data['label'].values)
    
    # Division des données
    train_inputs, test_inputs, train_masks, test_masks, train_labels, test_labels = train_test_split(
        input_ids, attention_mask, labels,
        test_size=test_size,
        random_state=42,
        stratify=labels
    )
    
    val_split = val_size / (1 - test_size)
    train_inputs, val_inputs, train_masks, val_masks, train_labels, val_labels = train_test_split(
        train_inputs, train_masks, train_labels,
        test_size=val_split,
        random_state=42,
        stratify=train_labels
    )
    
    # Création des DataLoaders
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    val_data = TensorDataset(val_inputs, val_masks, val_labels)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    test_data = TensorDataset(test_inputs, test_masks, test_labels)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    print("Données préparées avec succès.")
    print(f"Ensemble d'entraînement : {len(train_data)} échantillons")
    print(f"Ensemble de validation : {len(val_data)} échantillons")
    print(f"Ensemble de test : {len(test_data)} échantillons")
    
    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "tokenizer": tokenizer
    }
