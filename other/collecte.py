import cv2
import mediapipe as mp
import pandas as pd
import time

# Initialisation MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# CSV setup
csv_file = "gestes.csv"
colonnes = [f'x{i}' for i in range(42)] + [f'y{i}' for i in range(42)] + ['label']
try:
    pd.read_csv(csv_file)  # Juste pour vérifier que le fichier existe
except FileNotFoundError:
    pd.DataFrame(columns=colonnes).to_csv(csv_file, index=False)

cap = cv2.VideoCapture(0)

duree_entre_captures = 1  # secondes
nb_echantillons_par_geste = 10

print("→ Entrez le label souhaité (exemple : A, B, C ou 5, 10...)")
print("→ Positionnez vos deux mains après avoir saisi le label.")
print("→ Appuyez sur 'q' pour quitter à tout moment.\n")

while True:
    label = input("Tapez le label du geste : ")
    if not label:
        print("Label vide, fin de session.")
        break

    print(f"→ Prépare-toi pour capturer : '{label}'")
    sample_count = 0
    last_capture_time = time.time()

    while sample_count < nb_echantillons_par_geste:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        current_time = time.time()

        if result.multi_hand_landmarks and len(result.multi_hand_landmarks) == 2:
            all_coords = []
            for hand_landmarks in result.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    all_coords.append(lm.x)
                for lm in hand_landmarks.landmark:
                    all_coords.append(lm.y)

            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if len(all_coords) == 84 and current_time - last_capture_time >= duree_entre_captures:
                ligne = all_coords + [label]
                new_data = pd.DataFrame([ligne])
                new_data.to_csv(csv_file, mode='a', header=False, index=False)

                sample_count += 1
                last_capture_time = current_time
                print(f"✓ Capture {sample_count}/{nb_echantillons_par_geste} pour le label '{label}'")

                if sample_count == nb_echantillons_par_geste:
                    print(f"\n→ Label '{label}' terminé.\n")
                    time.sleep(1.5)

        cv2.putText(frame, f"Label: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
        cv2.putText(frame, f"Capture {sample_count+1}/{nb_echantillons_par_geste}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Appuie sur 'q' pour quitter", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Apprentissage Gestes", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            print("✓ Terminé. Toutes les données ont été enregistrées dans le fichier CSV.")
            exit()

cap.release()
cv2.destroyAllWindows()
print("✓ Terminé. Toutes les données ont été enregistrées dans le fichier CSV.")
