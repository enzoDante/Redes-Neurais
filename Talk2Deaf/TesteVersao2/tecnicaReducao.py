import json
import numpy as np

# Carregar o arquivo JSON
def carregar_gestos(arquivo_json):
    with open(arquivo_json, 'r') as arquivo:
        gestos = json.load(arquivo)
    return gestos

# Função para calcular a distância euclidiana entre dois pontos no espaço tridimensional
def calcular_distancia_3d(p1, p2):
    return np.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2 + (p1['z'] - p2['z'])**2)

# Função para remover frames redundantes com base na proximidade das coordenadas
def remover_frames_redundantes(frames, limite_distancia=0.01):
    frames_filtrados = [frames[0]]  # Manter sempre o primeiro frame
    for i in range(1, len(frames)):
        frame_anterior = frames_filtrados[-1]
        frame_atual = frames[i]
        
        distancias = []
        for key in frame_atual.keys():
            if key in frame_anterior:  # Garantir que a chave exista nos dois frames
                distancia = calcular_distancia_3d(frame_atual[key], frame_anterior[key])
                distancias.append(distancia)

        # Se a média das distâncias entre as partes do corpo for maior que o limite, mantemos o frame
        if np.mean(distancias) > limite_distancia:
            frames_filtrados.append(frame_atual)

    return frames_filtrados

# Função para aplicar truncamento ou padding, e remover frames redundantes
def ajustar_frames(gestos, num_frames_desejados, limite_distancia=0.01):
    gestos_ajustados = {}

    for rotulo, frames in gestos.items():
        #remover frames redundantes
        frames_filtrados = remover_frames_redundantes(frames, limite_distancia)

        if len(frames_filtrados) > num_frames_desejados:
            # Truncamento: Se tiver mais frames do que o necessário
            gestos_ajustados[rotulo] = frames_filtrados[:num_frames_desejados]
        elif len(frames_filtrados) < num_frames_desejados:
            # Padding: Se tiver menos frames, repetir frames até alcançar o número desejado
            frames_ajustados = frames_filtrados.copy()
            while len(frames_ajustados) < num_frames_desejados:
                frames_ajustados.append(frames_ajustados[-1])  # Adicionar o último frame repetido
            gestos_ajustados[rotulo] = frames_ajustados
        else:
            # Se já tiver o número exato de frames
            gestos_ajustados[rotulo] = frames_filtrados
    
    return gestos_ajustados

# Salvar os gestos ajustados em um novo arquivo JSON
def salvar_gestos_ajustados(arquivo_saida, gestos_ajustados):
    with open(arquivo_saida, 'w') as arquivo_json:
        json.dump(gestos_ajustados, arquivo_json, indent=4)

# Função principal
def processar_gestos(arquivo_entrada, arquivo_saida, num_frames_desejados, limite_distancia=0.01):
    # Carregar gestos do arquivo JSON
    gestos = carregar_gestos(arquivo_entrada)

    # Ajustar os frames (truncamento ou padding)
    gestos_ajustados = ajustar_frames(gestos, num_frames_desejados, limite_distancia)

    # Salvar os gestos ajustados em um novo arquivo
    salvar_gestos_ajustados(arquivo_saida, gestos_ajustados)
    print(f"Gestos ajustados salvos em {arquivo_saida}")

# Defina o número desejado de frames para cada sequência e o limite de distância
NUM_FRAMES_DESEJADOS = 30  # Você pode ajustar conforme necessário
LIMITE_DISTANCIA = 0.01 #Ajuste conforme necessário

# Chamar a função principal para processar os gestos
arquivo_entrada = 'gestos_coletados.json'
arquivo_saida = 'gestos_ajustados.json'
processar_gestos(arquivo_entrada, arquivo_saida, NUM_FRAMES_DESEJADOS, LIMITE_DISTANCIA)
