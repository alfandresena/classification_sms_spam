import torch
from importation import importer_donnees
from preprocessing import preparer_donnees
from roberta_model import RoBERTaModel

def main():
    # Définir le chemin du fichier
    chemin_fichier = "/content/classification_sms_spam/Data/spam.csv"
    
    # Configuration des hyperparamètres
    max_length = 128
    num_classes = 2
    epochs = 3
    learning_rate = 2e-5
    batch_size = 32
    
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
            max_length=max_length,
            batch_size=batch_size
        )
        
        if resultats:
            # Initialiser le modèle
            print("\nInitialisation du modèle RoBERTa...")
            model = RoBERTaModel(
                num_classes=num_classes,
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
            torch.save(model.model.state_dict(), 'spam_detector_roberta.pth')
            print("\nModèle sauvegardé avec succès dans 'spam_detector_roberta.pth'")
            
        else:
            print("Erreur lors de la préparation des données.")
    else:
        print("L'importation des données a échoué.")

if __name__ == "__main__":
    main()