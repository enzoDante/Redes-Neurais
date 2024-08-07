import cv2
import mediapipe as mp
import numpy as np
import csv

# Lista de rótulos (por exemplo, letras A, B, C...)
labels = ['A', 'B', 'C', ...]

def save_landmarks(landmarks, label):
    with open('hand_gestures.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(np.append(landmarks, label))

# Suponha que você tenha uma lógica para definir o rótulo atual
current_label = 'A'

def extract_hand_landmarks(results):
    landmarks = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])
    return np.array(landmarks).flatten() if landmarks else np.array([])


# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Inicializar captura de vídeo
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Converter imagem BGR para RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Processar imagem e detectar mãos
    result = hands.process(frame_rgb)
    
    # Desenhar landmarks nas mãos detectadas
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = extract_hand_landmarks(result)
            print(landmarks)  # Substituir isso com a lógica para salvar os dados
            if landmarks.size > 0:
                save_landmarks(landmarks, current_label)
    
    # Mostrar a imagem
    cv2.imshow('Hand Tracking', frame)
    
    if cv2.waitKey(1) & 0xFF == 27:  # Pressionar 'ESC' para sair
        break

cap.release()
cv2.destroyAllWindows()