import cv2
import mediapipe as mp
import json
import numpy as np

# Inicializando MediaPipe Hands
mp_hands = mp.solutions.hands
#hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

def extract_hand_landmarks(results):
    landmarks = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])
    return np.array(landmarks).flatten() if landmarks else np.array([])


# Dicionário para armazenar os gestos rotulados
labeled_gestures = {}

# Função para salvar a sequência de landmarks em um arquivo JSON
def save_labeled_gestures(labeled_gestures, file_path):
    with open(file_path, 'w') as f:
        json.dump(labeled_gestures, f, indent=4)

# Inicializar a captura de vídeo
cap = cv2.VideoCapture(0)

print("Pressione 'q' para parar a gravação, 'e' para escolher um rotulo ou 's' para salvar o rótulo atual.")
rotulo = "A"
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Converter a imagem para RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Processar a imagem e detectar as mãos
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            frame_landmarks = {}
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            #landmarks = extract_hand_landmarks(results)
            #if landmarks > 0:
            for idx, landmark in enumerate(hand_landmarks.landmark):
                frame_landmarks[idx] = {'x': landmark.x, 'y': landmark.y, 'z': landmark.z}

            # Solicitar o rótulo do gesto atual
            #rotulo = input("Digite o rótulo para este gesto: ")

            # Adicionar o frame ao rótulo correspondente
            if rotulo in labeled_gestures:
                labeled_gestures[rotulo].append(frame_landmarks)
            else:
                labeled_gestures[rotulo] = [frame_landmarks]

    # Mostrar o frame
    cv2.imshow('Frame', frame)

    # Pressionar 'q' para parar a gravação, ou 's' para salvar o rótulo atual
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('e'):
        rotulo = input("Digite o rótulo para este gesto: ")
    elif key == ord('s'):
        save_labeled_gestures(labeled_gestures, 'gestures_libras.json')
        print("Dados salvos!")

# Liberar a captura de vídeo e fechar janelas
cap.release()
cv2.destroyAllWindows()
hands.close()

# Salvar os dados finais
save_labeled_gestures(labeled_gestures, 'gestures_libras.json')
print("Gravação finalizada e dados salvos.")