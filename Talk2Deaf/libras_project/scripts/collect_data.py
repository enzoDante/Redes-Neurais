
import cv2
import mediapipe as mp
import numpy as np
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

# Diretório para salvar os dados
DATA_DIR = 'libras_project/data'

def collect_data():
    label = input("Digite o rótulo do gesto (ex: 'ola', 'obrigado'): ")
    num_sequences = int(input("Quantas sequências (amostras) você quer coletar para este gesto? "))
    frames_per_sequence = int(input("Quantos frames por sequência (gesto)? "))

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro: Não foi possível abrir a câmera.")
        return

    print(f"Preparando para coletar dados para o gesto: {label}")

    for sequence in range(num_sequences):
        print(f"\nColetando sequência {sequence + 1}/{num_sequences} para o gesto '{label}'...")
        sequence_data = []

        for frame_num in range(frames_per_sequence):
            ret, frame = cap.read()
            if not ret:
                print("Erro: Não foi possível ler o frame da câmera.")
                break

            # Inverte o frame horizontalmente para uma visualização espelhada
            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            # Desenha as anotações na imagem
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extrai as coordenadas dos pontos-chave
                keypoints = []
                for hand_landmarks in results.multi_hand_landmarks:
                    for landmark in hand_landmarks.landmark:
                        keypoints.append([landmark.x, landmark.y, landmark.z])
                sequence_data.append(keypoints)
            else:
                # Se nenhuma mão for detectada, adicione uma lista vazia ou zeros
                # Dependendo de como você quer tratar a ausência de detecção
                # Por simplicidade, vamos adicionar uma lista de zeros com o tamanho esperado
                # Isso pode ser ajustado para melhor atender ao treinamento da rede neural
                num_landmarks = 21 * 2 # 21 landmarks por mão, assumindo 2 mãos
                keypoints_placeholder = [[0.0, 0.0, 0.0]] * num_landmarks
                sequence_data.append(keypoints_placeholder)

            cv2.putText(frame, f'Gesto: {label} | Seq: {sequence+1}/{num_sequences} | Frame: {frame_num+1}/{frames_per_sequence}', 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('Coleta de Dados - Pressione Q para Sair', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        if len(sequence_data) == frames_per_sequence:
            # Salva a sequência como um arquivo .npy
            file_path = os.path.join(DATA_DIR, f'{label}_{sequence}.npy')
            np.save(file_path, np.array(sequence_data))
            print(f"Sequência salva em: {file_path}")
        else:
            print(f"Sequência {sequence + 1} incompleta. Não foi salva.")

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == '__main__':
    collect_data()


