import torch
from importation import importer_donnees
from preprocessing import preparer_donnees
from lstm_model import LSTMModel

def main():
    # Définir le chemin du fichier
    chemin_fichier = "../Data/spam.csv"
    
    # Configuration des hyperparamètres
    max_sequence_length = 100
    vocab_size = 5000
    embedding_dim = 100
    hidden_dim = 128
    num_layers = 2
    num_classes = 2
    dropout = 0.3
    epochs = 5
    learning_rate = 0.001
    
    # Vérifier si CUDA est disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation de : {device}")
    
    # Importer les données
    print("Importation des données...")
    donnees = importer_donnees(chemin_fichier)
    
    if donnees is not None:
        print("Aperçu des données :")
        print(donnees.head())
        
        # Prétraiter les données
        print("\nPréparation des données...")
        resultats = preparer_donnees(
            donnees,
            max_sequence_length=max_sequence_length,
            vocab_size=vocab_size
        )
        
        if resultats:
            # Initialiser le modèle
            print("\nInitialisation du modèle LSTM...")
            model = LSTMModel(
                input_dim=vocab_size,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_classes=num_classes,
                dropout=dropout,
                device=device
            )
            
            # Entraînement
            print("\nDébut de l'entraînement...")
            model.train(
                resultats["train_loader"],
                resultats["val_loader"],
                epochs=epochs,
                lr=learning_rate
            )
            
            # Évaluation
            print("\nÉvaluation du modèle sur l'ensemble de test...")
            test_f1 = model.evaluate(resultats["test_loader"])
            
            print(f"\nScore F1 final sur l'ensemble de test : {test_f1:.4f}")
            
            # Sauvegarder le modèle
            torch.save(model.model.state_dict(), 'spam_detector_lstm.pth')
            print("\nModèle sauvegardé avec succès dans 'spam_detector_lstm.pth'")
            
        else:
            print("Erreur lors de la préparation des données.")
    else:
        print("L'importation des données a échoué.")

if __name__ == "__main__":
    main()