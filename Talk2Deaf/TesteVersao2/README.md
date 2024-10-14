# Explicação:
## Função carregar_gestos:

- Carrega o arquivo JSON com os gestos coletados, que contém as coordenadas de cada gesto.

## Função calcular_distancia_3d:

- Calcula a distância euclidiana entre dois pontos tridimensionais (um ponto em um frame e o correspondente no próximo frame).
## Função remover_frames_redundantes:

- Remove frames cuja mudança nas coordenadas é menor do que um limite definido (limite_distancia).
- Mantém o primeiro frame e só adiciona outros frames se a média das distâncias entre as partes do corpo for maior que o limite.
## Truncamento/Padding:

- Após remover os frames redundantes, ainda aplicamos truncamento ou padding para garantir que todas as sequências tenham o mesmo número de frames.

### Função ajustar_frames:

- Ajusta a quantidade de frames para um valor fixo (num_frames_desejados).
- Se houver mais frames do que o valor desejado, os frames são truncados.
- Se houver menos frames, aplicamos padding duplicando o último frame até atingir o número necessário.

## Função salvar_gestos_ajustados:

- Salva as sequências ajustadas em um novo arquivo JSON (gestos_ajustados.json).

### Próximos Passos:

- Essa abordagem garante que todas as sequências tenham o mesmo tamanho, o que facilita o treinamento da rede neural.
## Parâmetros Ajustáveis:

- NUM_FRAMES_DESEJADOS: Número fixo de frames para cada gesto.
- LIMITE_DISTANCIA: Define quão "próximos" dois frames precisam estar para serem considerados redundantes. Você pode ajustar esse valor conforme necessário.
- Esse método ajuda a evitar a perda de frames importantes ao aplicar truncamento, removendo apenas aqueles que realmente são redundantes.