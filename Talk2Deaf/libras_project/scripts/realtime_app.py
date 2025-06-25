
import cv2
import mediapipe as mp
import numpy as np
import keras
import os

# Inicializa o MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Diretórios
MODELS_DIR = 'libras_project/models'

def preprocess_frame_data(frame_data, max_frames=30, num_landmarks=42, num_coords=3):
    # Redimensiona e normaliza os dados de um único frame para a entrada do modelo
    # Certifique-se de que esta função corresponda ao pré-processamento em train_model.py
    
    # Se o frame_data não tiver o número esperado de landmarks, preenche com zeros
    if len(frame_data) != num_landmarks:
        # Isso pode acontecer se apenas uma mão for detectada ou nenhuma
        # Para simplificar, vamos preencher com zeros para ter o tamanho esperado
        # Uma abordagem mais robusta pode ser necessária dependendo do modelo
        padded_frame_data = [[0.0, 0.0, 0.0]] * num_landmarks
        for i in range(min(len(frame_data), num_landmarks)):
            padded_frame_data[i] = frame_data[i]
        frame_data = padded_frame_data

    processed_frame = np.array(frame_data).reshape(1, num_landmarks, num_coords)
    
    # Normaliza as coordenadas (0-1)
    processed_frame = processed_frame / np.max(np.abs(processed_frame)) # Normaliza para -1 a 1 ou 0 a 1
    processed_frame = (processed_frame + 1) / 2 # Garante 0 a 1 se for -1 a 1

    # Achata as coordenadas para uma única dimensão por frame
    processed_frame = processed_frame.reshape(1, -1)

    # Para simular uma sequência de frames para o modelo LSTM, precisamos de `max_frames`
    # Aqui, estamos tratando cada frame como uma sequência de 1 para inferência em tempo real
    # ou acumulando frames para uma janela deslizante.
    # Para este exemplo, vamos considerar que o modelo espera uma sequência de `max_frames`
    # e usaremos uma janela deslizante.
    
    return processed_frame

def realtime_app():
    model_path = os.path.join(MODELS_DIR, 'libras_model.h5')
    classes_path = os.path.join(MODELS_DIR, 'classes.npy')

    if not os.path.exists(model_path):
        print(f"Erro: Modelo não encontrado em {model_path}. Por favor, treine o modelo primeiro.")
        return
    if not os.path.exists(classes_path):
        print(f"Erro: Arquivo de classes não encontrado em {classes_path}. Por favor, treine o modelo primeiro.")
        return

    model = keras.models.load_model(model_path)
    classes = np.load(classes_path)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro: Não foi possível abrir a câmera.")
        return

    print("Iniciando a aplicação de reconhecimento de Libras em tempo real...")

    # Buffer para armazenar os frames da sequência para inferência
    sequence_buffer = []
    max_frames = 30 # Deve ser o mesmo valor usado em train_model.py
    num_landmarks = 21 * 2 # 21 landmarks por mão, assumindo 2 mãos
    num_coords = 3 # x, y, z

    predicted_gesture = "Nenhum"

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro: Não foi possível ler o frame da câmera.")
            break

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        current_frame_keypoints = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                for landmark in hand_landmarks.landmark:
                    current_frame_keypoints.append([landmark.x, landmark.y, landmark.z])
        
        # Preenche com zeros se não houver detecção ou se o número de landmarks for diferente
        if len(current_frame_keypoints) < num_landmarks:
            current_frame_keypoints.extend([[0.0, 0.0, 0.0]] * (num_landmarks - len(current_frame_keypoints)))

        # Adiciona o frame pré-processado ao buffer da sequência
        processed_frame_for_buffer = preprocess_frame_data(current_frame_keypoints, num_landmarks=num_landmarks, num_coords=num_coords)
        sequence_buffer.append(processed_frame_for_buffer.flatten()) # Achata para adicionar ao buffer

        # Mantém o buffer com o tamanho máximo de frames
        if len(sequence_buffer) > max_frames:
            sequence_buffer.pop(0)

        # Realiza a inferência se o buffer estiver cheio
        if len(sequence_buffer) == max_frames:
            # Converte o buffer para um array numpy e redimensiona para a entrada do modelo
            input_sequence = np.array(sequence_buffer).reshape(1, max_frames, num_landmarks * num_coords)
            
            predictions = model.predict(input_sequence)
            predicted_class_index = np.argmax(predictions)
            predicted_gesture = classes[predicted_class_index]

        cv2.putText(frame, f'Gesto: {predicted_gesture}', 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Reconhecimento de Libras em Tempo Real - Pressione Q para Sair', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == '__main__':
    realtime_app()


