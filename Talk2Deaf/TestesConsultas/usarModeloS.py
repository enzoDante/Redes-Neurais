import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model

# Inicializa o MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Carrega o modelo
model = load_model('gesture_model.h5')

# Configurações
LABELS = label_encoder.classes_

def preprocess_frame(frame):
    # Normaliza e ajusta o formato do frame
    frame = (frame - np.mean(frame)) / np.std(frame)
    return np.expand_dims(frame, axis=0)

def predict_gesture(frame):
    processed_frame = preprocess_frame(frame)
    prediction = model.predict(processed_frame)
    return LABELS[np.argmax(prediction)]

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Converte a imagem para RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for landmarks in result.multi_hand_landmarks:
                # Obtém coordenadas das mãos
                hand_data = [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]
                
                # Previsão do gesto
                gesture_name = predict_gesture(np.array(hand_data))
                
                # Desenha a previsão na imagem
                cv2.putText(frame, gesture_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
        
        cv2.imshow('Gesture Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
