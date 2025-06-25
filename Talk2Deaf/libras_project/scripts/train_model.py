
import numpy as np
import os
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# Diretórios
DATA_DIR = 'libras_project/data'
MODELS_DIR = 'libras_project/models'

def load_data():
    sequences = []
    labels = []

    for filename in os.listdir(DATA_DIR):
        if filename.endswith('.npy'):
            label = filename.split('_')[0]  # Extrai o rótulo do nome do arquivo
            filepath = os.path.join(DATA_DIR, filename)
            data = np.load(filepath)
            sequences.append(data)
            labels.append(label)
    
    return np.array(sequences), np.array(labels)

def preprocess_data(sequences, labels):
    # Padroniza o número de frames por sequência (se necessário)
    # Por enquanto, vamos assumir que todas as sequências têm o mesmo número de frames
    # ou que a rede neural pode lidar com sequências de tamanhos variados (ex: LSTM)
    # Para este exemplo, vamos fazer um padding simples ou truncamento para um tamanho fixo
    # Você pode ajustar isso com base na sua coleta de dados e na complexidade dos gestos
    max_frames = 30 # Exemplo: defina um número máximo de frames
    num_landmarks = 21 * 2 # 21 landmarks por mão, assumindo 2 mãos
    num_coords = 3 # x, y, z

    processed_sequences = []
    for seq in sequences:
        if seq.shape[0] > max_frames:
            processed_sequences.append(seq[:max_frames, :, :]) # Trunca
        elif seq.shape[0] < max_frames:
            # Preenche com zeros
            padding = np.zeros((max_frames - seq.shape[0], num_landmarks, num_coords))
            processed_sequences.append(np.concatenate((seq, padding), axis=0))
        else:
            processed_sequences.append(seq)
    
    processed_sequences = np.array(processed_sequences)
    
    # Normaliza as coordenadas (0-1)
    # As coordenadas já vêm normalizadas pelo MediaPipe, mas uma normalização adicional pode ser útil
    # para garantir que os valores estejam dentro de um intervalo consistente para a rede neural.
    # Para este exemplo, vamos apenas garantir que os valores estejam entre 0 e 1.
    processed_sequences = processed_sequences / np.max(np.abs(processed_sequences)) # Normaliza para -1 a 1 ou 0 a 1
    processed_sequences = (processed_sequences + 1) / 2 # Garante 0 a 1 se for -1 a 1

    # Achata as coordenadas para uma única dimensão por frame
    # (frames, landmarks * coords)
    processed_sequences = processed_sequences.reshape(processed_sequences.shape[0], processed_sequences.shape[1], -1)

    # Codificação one-hot dos rótulos
    label_binarizer = LabelBinarizer()
    encoded_labels = label_binarizer.fit_transform(labels)

    return processed_sequences, encoded_labels, label_binarizer.classes_

def build_model(input_shape, num_classes):
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.LSTM(64, return_sequences=True),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(128),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model():
    sequences, labels = load_data()
    if len(sequences) == 0:
        print("Nenhum dado encontrado para treinamento. Por favor, execute collect_data.py primeiro.")
        return

    processed_sequences, encoded_labels, classes = preprocess_data(sequences, labels)

    X_train, X_test, y_train, y_test = train_test_split(processed_sequences, encoded_labels, test_size=0.2, random_state=42)

    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = len(classes)

    model = build_model(input_shape, num_classes)
    model.summary()

    print("\nIniciando o treinamento do modelo...")
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

    # Avalia o modelo
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nPrecisão do modelo no conjunto de teste: {accuracy*100:.2f}%")

    # Salva o modelo
    model_path = os.path.join(MODELS_DIR, 'libras_model.h5')
    model.save(model_path)
    print(f"Modelo salvo em: {model_path}")

    # Salva as classes para uso posterior na aplicação final
    classes_path = os.path.join(MODELS_DIR, 'classes.npy')
    np.save(classes_path, classes)
    print(f"Classes salvas em: {classes_path}")

if __name__ == '__main__':
    train_model()


