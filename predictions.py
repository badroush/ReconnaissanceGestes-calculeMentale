import cv2
import mediapipe as mp
import numpy as np
import joblib
import random
import csv
from tensorflow.keras.models import load_model
import pygame

# ==============================================
# SECTION INITIALISATION
# ==============================================

# Initialisation du système audio Pygame
pygame.mixer.init()
# Chargement des fichiers audio pour les feedbacks sonores
bravo_sound = pygame.mixer.Sound("bravo.mp3")  # Son pour réponse correcte
essaye_sound = pygame.mixer.Sound("essaye.mp3")  # Son pour réponse incorrecte

# Chargement du modèle et du scaler avec gestion d'erreur
try:
    model = load_model('modele_gestes.h5')  # Chargement du modèle Keras
    scaler = joblib.load('scaler_gestes.save')  # Chargement du normalisateur
except Exception as e:
    print(f"Erreur lors du chargement du modèle/scaler: {e}")
    exit(1)

# Configuration de MediaPipe pour la détection des mains
mp_hands = mp.solutions.hands
# Paramètres :
# - static_image_mode=False : mode vidéo (optimisé pour les flux)
# - max_num_hands=2 : détecte jusqu'à 2 mains
# - min_detection_confidence=0.8 : seuil de confiance minimal
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils  # Utilitaires pour dessiner les landmarks

# ==============================================
# FONCTIONS DE TRAITEMENT
# ==============================================

def extraire_caracteristiques(landmarks):
    """
    Extrait les coordonnées x et y des landmarks d'une main et les met à plat dans une liste.
    Args:
        landmarks: Objet contenant les points de repère de la main détectée
    Returns:
        Liste plate des coordonnées [x1, y1, x2, y2, ...]
    """
    return [coord for lm in landmarks for coord in (lm.x, lm.y)]

def normaliser_donnees(caracteristiques, scaler):
    """
    Normalise les caractéristiques extraites à l'aide du scaler pré-entraîné.
    Args:
        caracteristiques: Liste des caractéristiques à normaliser
        scaler: Normalisateur pré-entraîné
    Returns:
        Caractéristiques normalisées
    """
    return scaler.transform([caracteristiques])[0]

def prediction_main(frame, model, scaler):
    """
    Effectue la prédiction du geste à partir d'une image.
    Args:
        frame: Image capturée depuis la webcam
        model: Modèle de classification pré-entraîné
        scaler: Normalisateur pour les caractéristiques
    Returns:
        Tuple contenant:
        - L'image avec annotations (si détection)
        - La prédiction (ou None si pas de détection)
        - Les caractéristiques brutes (pour sauvegarde)
    """
    # Conversion de l'espace colorimétrique BGR à RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Détection des mains avec MediaPipe
    result = hands.process(rgb)
    
    # Vérification qu'on a bien détecté exactement 2 mains
    if result.multi_hand_landmarks and len(result.multi_hand_landmarks) == 2:
        donnees_gauche, donnees_droite = None, None
        
        # Séparation des données main gauche/droite
        for landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            label = handedness.classification[0].label
            if label == 'Left':
                donnees_gauche = landmarks
            elif label == 'Right':
                donnees_droite = landmarks
        
        # Si les deux mains sont détectées
        if donnees_gauche and donnees_droite:
            # Extraction des caractéristiques des deux mains (84 valeurs : 21 landmarks * 2 coordonnées * 2 mains)
            features = extraire_caracteristiques(donnees_gauche.landmark) + extraire_caracteristiques(donnees_droite.landmark)
            
            # Vérification du nombre de caractéristiques
            if len(features) != 84:
                return frame, None, None
            
            # Normalisation et prédiction
            features_scaled = normaliser_donnees(features, scaler)
            prediction = model.predict(np.reshape(features_scaled, (1, 84)), verbose=0)
            
            return frame, np.argmax(prediction, axis=1)[0], features
    
    return frame, None, None

# ==============================================
# FONCTIONS DU JEU
# ==============================================

def generer_operation():
    """
    Génère une opération mathématique aléatoire avec son résultat.
    Returns:
        Tuple contenant:
        - Chaîne représentant l'opération (ex: "5 + 3")
        - Résultat de l'opération
    """
    while True:
        a, b = random.randint(0, 99), random.randint(0, 99)
        op = random.choice(['+', '-'])  # Choix aléatoire entre addition et soustraction
        resultat = a + b if op == '+' else a - b
        
        # On s'assure que le résultat est entre 0 et 99
        if 0 <= resultat <= 99:
            return f"{a} {op} {b}", resultat

def afficher_countdown(frame, seconds):
    """
    Affiche un compte à rebours sur l'image.
    Args:
        frame: Image sur laquelle afficher le compte à rebours
        seconds: Durée du compte à rebours en secondes
    """
    h, w, _ = frame.shape
    for sec in range(seconds, 0, -1):
        temp = frame.copy()
        texte = f"Nouvelle operation dans : {sec}"
        # Calcul de la position du texte pour le centrer
        text_size = cv2.getTextSize(texte, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        text_x = (w - text_size[0]) // 2
        text_y = h - 30
        # Affichage du texte
        cv2.putText(temp, texte, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 3)
        cv2.imshow("Calcul Mental Gestuel", temp)
        # Attente d'une seconde ou jusqu'à appui sur 'q'
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break

def sauvegarder_donnees(features, expected_value):
    """
    Sauvegarde les caractéristiques et le résultat attendu dans un fichier CSV.
    Args:
        features: Caractéristiques à sauvegarder
        expected_value: Valeur attendue (bonne réponse)
    """
    with open("gestes.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(features + [expected_value])

# ==============================================
# BOUCLE PRINCIPALE DU JEU
# ==============================================

# Initialisation de la capture vidéo
cap = cv2.VideoCapture(0)

# Génération de la première opération
operation, solution = generer_operation()

# Variables d'état du jeu
dernier_resultat = ""  # Dernier message affiché
score = 0              # Score du joueur
tentatives = 0         # Nombre de tentatives pour l'opération actuelle

# Boucle principale
while True:
    # Capture d'une frame depuis la webcam
    ret, frame = cap.read()
    if not ret:
        break
    
    # Miroir de l'image pour un effet plus naturel
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Affichage de l'opération en cours
    cv2.putText(frame, f"{operation} = ?", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 0), 3)

    # Prédiction du geste
    frame, predicted, features = prediction_main(frame, model, scaler)

    # Si un geste est détecté
    if predicted is not None:
        # Affichage de la prédiction
        cv2.putText(frame, f"Votre reponse: {predicted}", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 165, 255), 2)

        # Réponse correcte
        if predicted == solution:
            dernier_resultat = "✅ Bravo !"
            score += solution  # Augmentation du score
            tentatives = 0
            pygame.mixer.Sound.play(bravo_sound, 0, 2000, 200)  # Son de succès

            # Affichage du message de succès
            temp = frame.copy()
            cv2.putText(temp, dernier_resultat, (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
            cv2.putText(temp, f"Score : {score}", (w - 200, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 0, 128), 3)
            cv2.imshow("Calcul Mental Gestuel", temp)
            cv2.waitKey(2000)  # Pause de 2 secondes

            # Compte à rebours avant nouvelle opération
            afficher_countdown(temp, 3)
            operation, solution = generer_operation()
            dernier_resultat = ""

        # Réponse incorrecte
        else:
            tentatives += 1
            dernier_resultat = "❌ Essaye encore !"
            pygame.mixer.Sound.play(essaye_sound)  # Son d'échec

            # Affichage du message d'échec
            temp = frame.copy()
            cv2.putText(temp, dernier_resultat, (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
            cv2.putText(temp, f"Score : {score}", (w - 200, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 0, 128), 3)
            cv2.imshow("Calcul Mental Gestuel", temp)
            cv2.waitKey(2000)  # Pause de 2 secondes

            # Après 5 tentatives échouées
            if tentatives >= 5:
                # Sauvegarde des données pour améliorer le modèle
                if features:
                    sauvegarder_donnees(features, solution)
                    print(f"🚨 Données sauvegardées — attendu : {solution}")
                
                score = max(0, score - solution)  # Pénalité
                afficher_countdown(temp, 3)  # Compte à rebours
                operation, solution = generer_operation()  # Nouvelle opération
                tentatives = 0
                dernier_resultat = ""

    # Si aucun geste détecté mais message à afficher
    else:
        if dernier_resultat:
            color = (0, 255, 0) if "Bravo" in dernier_resultat else (0, 0, 255)
            cv2.putText(frame, dernier_resultat, (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)

    # Affichage du score en permanence
    cv2.putText(frame, f"Score : {score}", (w - 200, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 0, 128), 3)
    
    # Affichage de la frame
    cv2.imshow("Calcul Mental Gestuel", frame)
    
    # Sortie si 'q' est pressé
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Nettoyage
cap.release()
cv2.destroyAllWindows()