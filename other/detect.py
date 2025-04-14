import cv2
import mediapipe as mp
import csv
import os

# Initialisation MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

# Fichier CSV
csv_file = 'gestes.csv'
header = [f'L_x{i}' for i in range(21)] + [f'L_y{i}' for i in range(21)] + \
         [f'R_x{i}' for i in range(21)] + [f'R_y{i}' for i in range(21)] + ['label']

# Crée le fichier si non existant
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

# Calcule la valeur des doigts levés pour chaque main
def valeur_main_gauche(landmarks):
    valeur = 0
    lm = landmarks.landmark  # récupère la liste des 21 points
    if lm[4].x > lm[2].x:
        valeur += 50
    for tip in [8, 12, 16, 20]:
        if lm[tip].y < lm[tip - 2].y:
            valeur += 10
    return valeur
def valeur_main_droite(landmarks):
    valeur = 0
    lm = landmarks.landmark
    if lm[4].x < lm[2].x:
        valeur += 5
    for tip in [8, 12, 16, 20]:
        if lm[tip].y < lm[tip - 2].y:
            valeur += 1
    return valeur
# Capture vidéo
cap = cv2.VideoCapture(0)
print("Appuyez sur 'q' pour quitter")

enregistrement_actif = True

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks and len(result.multi_hand_landmarks) == 2:
        donnees_gauche, donnees_droite = None, None
        val_gauche, val_droite = 0, 0

        # Associer les mains par label
        for landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            label = handedness.classification[0].label
            if label == 'Left':
                donnees_gauche = landmarks
                val_gauche = valeur_main_gauche(landmarks)
            elif label == 'Right':
                donnees_droite = landmarks
                val_droite = valeur_main_droite(landmarks)

        # Si les deux mains sont bien détectées
        if donnees_gauche and donnees_droite:
            ligne = []
            for lm in donnees_gauche.landmark:
                ligne.extend([lm.x, lm.y])
            for lm in donnees_droite.landmark:
                ligne.extend([lm.x, lm.y])
            total = val_gauche + val_droite
            ligne.append(total)

            # Enregistrer dans le CSV
            with open(csv_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(ligne)

            # Afficher à l'écran
            cv2.putText(frame, f"Valeur: {total}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Dessiner les mains
        for lm in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Capture des gestes", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
