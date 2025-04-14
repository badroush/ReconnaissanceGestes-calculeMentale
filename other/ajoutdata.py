import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Charger le fichier CSV existant
data = pd.read_csv('gestes.csv')

# Supposons que les colonnes X1, Y1, ..., Xn, Yn représentent les coordonnées des doigts pour chaque geste
scaler = MinMaxScaler()
X = data.iloc[:, :-1].values  # Données des gestes
y = data.iloc[:, -1].values   # Labels des chiffres (1-99)

# Normaliser les données (si nécessaire)
X_scaled = scaler.fit_transform(X)

# Fonction pour appliquer des transformations simples : translation, rotation, bruit
def augmenter_donnees(X, y, augmentation_factor=0.1):
    X_augmented = []
    y_augmented = []
    
    for i in range(len(X)):
        # Ajouter du bruit léger
        noise = np.random.normal(0, augmentation_factor, X[i].shape)
        new_geste = X[i] + noise
        
        # Appliquer une légère translation (décalage des points de coordonnées)
        translation = np.random.uniform(-augmentation_factor, augmentation_factor, X[i].shape)
        new_geste = new_geste + translation
        
        # Appliquer une légère rotation (par exemple une rotation de 5 degrés)
        # Vérifier que la forme est correcte
        if new_geste.shape[0] == 2:  # Vérifier si les coordonnées sont 2D
            angle = np.random.uniform(-5, 5)
            rotation_matrix = np.array([
                [np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
                [np.sin(np.radians(angle)), np.cos(np.radians(angle))]
            ])
            new_geste = np.dot(new_geste.reshape(-1, 2), rotation_matrix)  # Appliquer la rotation aux coordonnées
        else:
            # Si les coordonnées ne sont pas en 2D, ajustez ce bloc en fonction de la structure de vos données.
            pass
        
        X_augmented.append(new_geste)
        y_augmented.append(y[i])
    
    return np.array(X_augmented), np.array(y_augmented)

# Calculer combien d'itérations sont nécessaires pour dépasser 20 000 lignes
target_rows = 4000
current_rows = len(X_scaled)
augmentation_factor = 0.1

# Calculer le nombre d'itérations nécessaires
num_iterations = (target_rows // current_rows) + 1
print(f"Augmentations nécessaires pour atteindre {target_rows} lignes: {num_iterations}")

# Générer les données augmentées
X_combined = X_scaled.copy()
y_combined = y.copy()

for _ in range(num_iterations):
    X_new, y_new = augmenter_donnees(X_combined, y_combined, augmentation_factor=augmentation_factor)
    X_combined = np.vstack([X_combined, X_new])
    y_combined = np.concatenate([y_combined, y_new])

# S'assurer que nous avons plus de 20 000 lignes
print(f"Forme finale de X_combined : {X_combined.shape}")
print(f"Forme finale de y_combined : {y_combined.shape}")

# Inverser la normalisation pour obtenir des valeurs originales (si nécessaire)
X_final = scaler.inverse_transform(X_combined)

# Sauvegarder les nouvelles données dans un nouveau fichier CSV
new_data = pd.DataFrame(X_final, columns=[f'X{i+1}' for i in range(X_final.shape[1])])
new_data['Label'] = y_combined
new_data.to_csv('gestes.csv', index=False)

print("Les données ont été augmentées et sauvegardées dans 'gestes.csv'.")
