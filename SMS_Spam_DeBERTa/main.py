import torch
from importation import importer_donnees
from preprocessing import preparer_donnees
from deberta_model import DeBERTaModel

def main():
    # Configuration
    chemin_fichier = "/kaggle/working/classification_sms_spam/Data/spam.csv"
    max_length = 128
    batch_size = 16
    num_classes = 2
    epochs = 3
    learning_rate = 2e-5
    
    # Vérification du device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation de : {device}")
    
    # Importation des données
    print("Importation des données...")
    donnees = importer_donnees(chemin_fichier)
    
    if donnees is not None:
        print("Aperçu des données :")
        print(donnees.head())
        
        # Prétraitement
        print("\nPréparation des données...")
        resultats = preparer_donnees(
            donnees,
            max_length=max_length,
            batch_size=batch_size
        )
        
        if resultats:
            # Initialisation du modèle
            print("\nInitialisation du modèle DeBERTa...")
            model = DeBERTaModel(num_classes=num_classes, device=device)
            
            # Entraînement
            print("\nDébut de l'entraînement...")
            model.train(
                resultats["train_loader"],
                resultats["val_loader"],
                epochs=epochs,
                lr=learning_rate
            )
            
            # Évaluation finale
            print("\nÉvaluation du modèle sur l'ensemble de test...")
            test_loss, test_f1 = model.evaluate(resultats["test_loader"])
            print(f"\nRésultats finaux sur l'ensemble de test:")
            print(f"  Loss: {test_loss:.4f}")
            print(f"  F1 Score: {test_f1:.4f}")
            
            # Sauvegarde du modèle
            torch.save(model.model.state_dict(), 'spam_detector_deberta.pth')
            print("\nModèle sauvegardé avec succès dans 'spam_detector_deberta.pth'")
        else:
            print("Erreur lors de la préparation des données.")
    else:
        print("L'importation des données a échoué.")

if __name__ == "__main__":
    main()