import cv2
import mediapipe as mp
import json

# Inicializando MediaPipe Hands e Pose
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Função para coletar coordenadas
def coletar_coordenadas(resultados_hands, resultados_pose):
    coordenadas = {}

    # Coletar coordenadas das mãos, se visíveis
    if resultados_hands.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(resultados_hands.multi_hand_landmarks):
            mao = f"mao{idx+1}"
            coordenadas[mao] = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand_landmarks.landmark]

    # Coletar coordenadas de ombros e cotovelos, se visíveis
    if resultados_pose.pose_landmarks:
        pose_landmarks = resultados_pose.pose_landmarks.landmark
        coordenadas["ombroEsquerdo"] = {"x": pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, "y": pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y, "z": pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].z}
        coordenadas["ombroDireito"] = {"x": pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, "y": pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y, "z": pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].z}
        coordenadas["cotoveloEsquerdo"] = {"x": pose_landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, "y": pose_landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y, "z": pose_landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].z}
        coordenadas["cotoveloDireito"] = {"x": pose_landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, "y": pose_landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y, "z": pose_landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].z}

    return coordenadas

# Função para salvar as coordenadas no arquivo JSON
def salvar_gestos_em_json(nome_arquivo, dados_gestos):
    with open(nome_arquivo, 'w') as arquivo_json:
        json.dump(dados_gestos, arquivo_json, indent=4)

# Função principal para coleta de gestos
def coletar_gestos():
    cap = cv2.VideoCapture(0)  # Abrir a câmera

    gestos = {}  # Dicionário para armazenar gestos
    gravando = False  # Flag para saber se está coletando gestos
    rotulo_gesto = ""  # Rótulo do gesto atual
    frames_gesto = []  # Armazenar os frames do gesto atual

    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands, \
         mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Erro ao acessar a câmera.")
                break

            # Converter BGR para RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False

            # Processar mãos e pose
            resultados_hands = hands.process(image_rgb)
            resultados_pose = pose.process(image_rgb)

            # Converter de volta para BGR
            image_rgb.flags.writeable = True
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            # Desenhar os landmarks
            if resultados_hands.multi_hand_landmarks:
                for hand_landmarks in resultados_hands.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if resultados_pose.pose_landmarks:
                mp_drawing.draw_landmarks(image_bgr, resultados_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Mostrar imagem com landmarks
            cv2.imshow('Coleta de Gestos', image_bgr)

            # Se estiver gravando, coletar coordenadas
            if gravando:
                coordenadas_frame = coletar_coordenadas(resultados_hands, resultados_pose)
                if coordenadas_frame:
                    frames_gesto.append(coordenadas_frame)

            # Capturar entrada do teclado
            key = cv2.waitKey(1) & 0xFF

            # Pressionar 'Enter' (13) para iniciar/parar a gravação
            if key == 13:  # Enter
                if not gravando:
                    print(f"Iniciando a gravação do gesto: {rotulo_gesto}")
                    frames_gesto = []
                    gravando = True
                else:
                    print(f"Parando a gravação do gesto: {rotulo_gesto}")
                    gravando = False

            # Pressionar 'Espaço' (32) para salvar a sequência e iniciar um novo rótulo
            if key == 32 and not gravando:  # Espaço
                if frames_gesto:
                    if rotulo_gesto in gestos:
                        gestos[rotulo_gesto].extend(frames_gesto)
                    else:
                        gestos[rotulo_gesto] = frames_gesto

                    print(f"Gesto '{rotulo_gesto}' salvo com {len(frames_gesto)} frames.")
                
                # Pedir um novo rótulo
                rotulo_gesto = input("Digite o rótulo do próximo gesto (ou 'sair' para finalizar): ")
                if rotulo_gesto.lower() == 'sair':
                    break
                frames_gesto = []  # Limpar a lista de frames para o próximo gesto

    # Finalizar e salvar os dados em um arquivo JSON
    salvar_gestos_em_json("gestos_coletados.json", gestos)
    cap.release()
    cv2.destroyAllWindows()

# Executar coleta de gestos
coletar_gestos()

#1 - aperte espaço para digitar o rotulo de um gesto
#2 - aperte enter para gravar esses gestos
#3 - aperte enter para parar a gravação dos gestos
#4 - aperte espaço para salvar esses gestos gravados
#5 - criar arquivo para usar as técnicas e modificar o arquivo com os gestos