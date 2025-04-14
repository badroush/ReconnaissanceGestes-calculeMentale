import numpy as np 
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib  # Pour sauvegarder le scaler

# Charger les données CSV
data = pd.read_csv('gestes.csv')
print("Données chargées avec succès.")

# Séparer X et y
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
print("Séparation des données réussie.")

# Normalisation des données avec MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)  

# Mapper les labels vers une séquence d'entiers
unique_labels = np.unique(y)
label_to_index = {label: index for index, label in enumerate(unique_labels)}
y_mapped = np.array([label_to_index[val] for val in y])
num_classes = len(unique_labels)

# Encoder les labels en one-hot
y_categorical = to_categorical(y_mapped, num_classes=num_classes)

# Diviser en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_categorical, test_size=0.2, random_state=42)

# Création du modèle
model = Sequential()
model.add(Dense(256, input_shape=(X_train.shape[1],), activation='relu'))  # Couche d'entrée
model.add(Dense(128, activation='relu'))  # Couche cachée
model.add(Dropout(0.3))  # Dropout pour éviter le surapprentissage
model.add(Dense(num_classes, activation='softmax'))  # Couche de sortie avec softmax pour la classification multi-classes

# Compilation du modèle
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entraînement du modèle
model.fit(X_train, y_train, epochs=250, batch_size=32, validation_split=0.2)
print("Entraînement terminé.")
print("score d'entraînement : ", model.evaluate(X_train, y_train)[1])
print("score de test : ", model.evaluate(X_test, y_test)[1])
print("occurency : ", np.bincount(y_mapped))

# Sauvegarder le modèle et le scaler
model.save('modele_gestes.h5')
joblib.dump(scaler, 'scaler_gestes.save')
print("Modèle et scaler sauvegardés avec succès.")
