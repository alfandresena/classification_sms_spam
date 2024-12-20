from transformers import DebertaTokenizer
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np

def preparer_donnees(data, max_length=128, test_size=0.2, val_size=0.1, batch_size=32):
    """
    Prépare les données pour l'entraînement avec DeBERTa.
    
    Args:
        data (pd.DataFrame): Le dataframe contenant les données.
        max_length (int): Longueur maximale des séquences.
        test_size (float): Proportion des données pour le test.
        val_size (float): Proportion des données pour la validation.
        batch_size (int): Taille des lots pour l'entraînement.
    
    Returns:
        dict: Contient les loaders et le tokenizer.
    """
    # Nettoyage des données
    data = data[['v1', 'v2']].dropna()
    data.columns = ['label', 'message']
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})
    
    # Initialisation du tokenizer DeBERTa
    tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
    
    # Tokenisation
    encodages = tokenizer(
        data['message'].tolist(),
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    
    # Préparation des données
    input_ids = encodages['input_ids']
    attention_mask = encodages['attention_mask']
    labels = torch.tensor(data['label'].values)
    
    # Division des données
    X_train_idx, X_test_idx, y_train, y_test = train_test_split(
        np.arange(len(labels)),
        labels,
        test_size=test_size,
        stratify=labels,
        random_state=42
    )
    
    val_size_adj = val_size / (1 - test_size)
    X_train_idx, X_val_idx, y_train, y_val = train_test_split(
        X_train_idx,
        y_train,
        test_size=val_size_adj,
        stratify=y_train,
        random_state=42
    )
    
    # Création des DataLoaders
    train_dataset = TensorDataset(
        input_ids[X_train_idx],
        attention_mask[X_train_idx],
        y_train
    )
    val_dataset = TensorDataset(
        input_ids[X_val_idx],
        attention_mask[X_val_idx],
        y_val
    )
    test_dataset = TensorDataset(
        input_ids[X_test_idx],
        attention_mask[X_test_idx],
        y_test
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    print(f"Taille de l'ensemble d'entraînement : {len(X_train_idx)}")
    print(f"Taille de l'ensemble de validation : {len(X_val_idx)}")
    print(f"Taille de l'ensemble de test : {len(X_test_idx)}")
    
    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "tokenizer": tokenizer
    }