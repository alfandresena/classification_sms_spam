import torch
from importation import importer_donnees
from preprocessing import preparer_donnees
from bert_model import BertModel

def main():
    # Configuration
    chemin_fichier = "/content/classification_sms_spam/Data/spam.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparamètres
    num_classes = 2
    batch_size = 16
    epochs = 3
    learning_rate = 2e-4
    
    print(f"Utilisation de : {device}")
    
    # 1. Importation des données
    print("\nImportation des données...")
    donnees = importer_donnees(chemin_fichier)
    
    if donnees is None:
        print("Erreur lors de l'importation des données.")
        return
    
    # 2. Prétraitement des données
    print("\nPréparation des données...")
    resultats = preparer_donnees(donnees, batch_size=batch_size)
    
    # 3. Initialisation et entraînement du modèle
    print("\nInitialisation du modèle BERT...")
    model = BertModel(num_classes=num_classes, device=device)
    
    print("\nDébut de l'entraînement...")
    model.train(
        resultats["train_loader"],
        resultats["val_loader"],
        epochs=epochs,
        lr=learning_rate
    )
    
    # 4. Évaluation finale
    print("\nÉvaluation finale du modèle...")
    test_f1 = model.evaluate(resultats["test_loader"])
    
    # 5. Sauvegarde du modèle
    torch.save(model.model.state_dict(), 'spam_detector_bert.pth')
    print("\nModèle sauvegardé avec succès dans 'spam_detector_bert.pth'")

if __name__ == "__main__":
    main()