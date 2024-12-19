import pandas as pd

def importer_donnees(chemin_fichier):
    """
    Importe les données d'un fichier CSV avec gestion de différents encodages.
    
    Args:
        chemin_fichier (str): Chemin vers le fichier CSV à importer.
    
    Returns:
        pandas.DataFrame: Le dataframe contenant les données importées.
    """
    encodages_possibles = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']  # Ajoutez d'autres encodages si nécessaire
    
    for encodage in encodages_possibles:
        try:
            donnees = pd.read_csv(chemin_fichier, encoding=encodage)
            print(f"Données importées avec succès depuis {chemin_fichier} avec l'encodage {encodage}")
            return donnees
        except UnicodeDecodeError:
            print(f"Échec avec l'encodage {encodage}, on essaie le suivant...")
        except Exception as e:
            print(f"Erreur inattendue : {e}")
    
    print(f"Impossible de lire le fichier {chemin_fichier}.")
    return None
