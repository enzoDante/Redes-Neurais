import cv2
import numpy as np
import keras
import mediapipe as mp
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

# Carrega o modelo
model = keras.models.load_model('C:\\Users\\Souls Dante\\Documents\\vsCode\\Redes-Neurais\\Talk2Deaf\\modelo_2\\modelo2.h5')


# Inicializa o MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


# Configurações
sequence_length = 10
window = []


def preprocess_frame(frame):
    # Normaliza e ajusta o formato do frame
    frame = (frame - np.mean(frame)) / np.std(frame)
    return frame


def predict_gesture(sequence):
    processed_sequence = np.expand_dims(sequence, axis=0)
    prediction = model.predict(processed_sequence)
    return label_encoder.classes_[np.argmax(prediction)]


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
                hand_data = np.array([(lm.x, lm.y, lm.z) for lm in landmarks.landmark])
               
                # Adiciona o frame ao final da janela
                window.append(hand_data)
               
                # Remove o frame mais antigo se a janela estiver cheia
                if len(window) > sequence_length:
                    window.pop(0)
               
                # Faz a predição se a janela estiver cheia
                if len(window) == sequence_length:
                    gesture_name = predict_gesture(np.array(window))
                    cv2.putText(frame, gesture_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
               
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
       
        cv2.imshow('Gesture Recognition', frame)
       
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
