import math
import cv2
import mediapipe as mp
import numpy as np

def main():
    # Inizializzazione di Mediapipe Hands e setup della webcam
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1)
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Errore nell'aprire la webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Ottieni le dimensioni del frame
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    print("Premi 'q' per uscire.")

    # Variabili per il disegno
    started = False
    prev = None  # Coordinate precedenti

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Errore nel leggere il frame dalla webcam")
            break

        # Capovolgimento del frame orizzontalmente per una vista speculare
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Elaborazione del frame con Mediapipe Hands
        results = hands.process(rgb_frame)

        MIN_DIST = 0.07  # Distanza minima per attivare il disegno

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Disegna i landmarks della mano sul frame
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Landmark della punta delle dita
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

                # Coordinate normalizzate
                hand = {
                    "x": (index_tip.x + middle_tip.x + thumb_tip.x) / 3,
                    "y": (index_tip.y + middle_tip.y + thumb_tip.y) / 3
                }

                # Distanze tra il centro della mano e le dita
                thumb_dist = math.sqrt((thumb_tip.x - hand["x"]) ** 2 + (thumb_tip.y - hand["y"]) ** 2)
                middle_dist = math.sqrt((middle_tip.x - hand["x"]) ** 2 + (middle_tip.y - hand["y"]) ** 2)
                index_dist = math.sqrt((index_tip.x - hand["x"]) ** 2 + (index_tip.y - hand["y"]) ** 2)

                # Logica per attivare il disegno
                if thumb_dist < MIN_DIST and middle_dist < MIN_DIST and index_dist < MIN_DIST:
                    cv2.circle(frame, (int(hand["x"] * width), int(hand["y"] * height)), 10, (0, 255, 0), -1)
                    if not started:
                        started = True
                        prev = hand  # Salva il punto iniziale
                    else:
                        # Disegna la linea tra il punto precedente e quello attuale
                        cv2.line(
                            canvas,
                            (int(prev["x"] * width), int(prev["y"] * height)),
                            (int(hand["x"] * width), int(hand["y"] * height)),
                            (255, 0, 0),
                            10
                        )
                        prev = hand  # Aggiorna il punto precedente
                else:
                    # Indica che il disegno è disattivato
                    cv2.circle(frame, (int(hand["x"] * width), int(hand["y"] * height)), 10, (0, 0, 255), -1)
                    started = False
        else:
            started = False

        # Combina il frame e la canvas
        combined = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

        # Mostra il risultato
        cv2.imshow("Drawing", combined)

        # Premi 'q' per uscire
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if cv2.waitKey(1) & 0xFF == ord('r'):
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
    # Rilascia risorse
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
