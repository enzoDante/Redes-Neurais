import cv2
import mediapipe as mp
import json
import os
#quando rodar o programa, ele coleta todos os frames, ao apertar btn de salvar, ele salva os frames em um arquivo


# Inicializa o MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


# Configurações
DATA_FILE = 'C:\\Users\\Souls Dante\\Documents\\vsCode\\Redes-Neurais\\Talk2Deaf\\modelo_2\\gestos2.json'


# Carrega dados existentes
if os.path.exists(DATA_FILE):
    with open(DATA_FILE, 'r') as file:
        data = json.load(file)
else:
    data = {}


def save_data():
    with open(DATA_FILE, 'w') as file:
        json.dump(data, file, indent=4)


def capture_gestures():
    cap = cv2.VideoCapture(0)
    gesture_name = None
    recording = False
    frames = []


    while True:
        ret, frame = cap.read()
        if not ret:
            break
       
        # Converte a imagem para RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)


        if result.multi_hand_landmarks:
            for landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
               
                if recording:
                    frame_data = [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]
                    frames.append(frame_data)
                    cv2.putText(frame, "Gravando...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Gesture Recorder', frame)
       
        key = cv2.waitKey(1)
        if key == ord('q'):  # Press 'q' to quit
            break
        elif key == ord('r'):  # Press 'r' to start/stop recording tem q apertar 2 vezes p salvar
            if recording:
                # Salva o gesto
                if gesture_name:
                    data[gesture_name] = frames
                    save_data()
                gesture_name = None
                recording = False
                frames = []
            else:
                gesture_name = input('Digite o rótulo do gesto: ')
                recording = True
        elif key == ord('s'):  # Press 's' to save data
            save_data()
   
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_gestures()
