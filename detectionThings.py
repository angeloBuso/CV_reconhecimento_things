"""

Projeto em que dado uma captura realizada e usando uma NN treinada consegue-se detectar objetos
Implementação da arquitetura YOLO baseada no artigo "YOLO object detection with OpenCV"
do Adrian Rosebrock, autor do site PyImageSearch
"""

# importar os pacotes necessários
import numpy as np
import argparse
import os
import cv2
import time
from imutils.video import VideoStream, FileVideoStream
from imutils.video import FPS

"""
Declaração de Variáveis Globais são necessárias quando trabalha-se com DeepLearning.
CONFIDENCE_MIN = threshold minimo de confinaça da previsão realizada
MODEL_BASE_PATH - é onde indicamos o modelo de deepLearning treinado. Yolo é um modelo deep treinado que pode detectar objetos
NN treinada permite (i.) flexibilidade e agilidade para o projeto (ii.) já temos os pesos "forward Propagation da NN"
"""

CONFIDENCE_MIN = 0.4
NMS_THRESHOLD = 0.2
MODEL_BASE_PATH = "yolo-coco"

# 1. Receber argumentos de entrada para o script
"""
Usando a função "argument" quando da chamada da execução do script deverá passar como argumento:
a) se captura Real Time       : -i IpSeverRTMP/password
b) se video descarregado local: -i E:\Portifolio\DRAFT_Pycharm\CV_detectionThings\IMG_0315.mp4
"""
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="Endereço do streaming do drone")
streaming_path = vars(ap.parse_args())['input']

# 2. Extrair os nomes das classes a partir do arquivo da NN Treinada (coco.names)
print("[+] Carregando labels das classes treinadas...")
with open(os.path.sep.join([MODEL_BASE_PATH, 'coco.names'])) as f:
    labels = f.read().strip().split("\n")

    # 2.1 Gerando cores únicas para cada label - tipo de objeto identificado
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

# 3. Carregando a Rede Neural (NN - modelo treinado YOLO (c/ COCO dataset)) já treinada (dois parametros: config, modelo/pesos)
print("[+] Carregando o modelo YOLO treinado no COCO dataset...")
net = cv2.dnn.readNetFromDarknet(
    os.path.sep.join([MODEL_BASE_PATH, 'yolov3.cfg']),
    os.path.sep.join([MODEL_BASE_PATH, 'yolov3.weights']))

# 3.1 Extrair layers (camadas) não conectados da arquitetura YOLO (default para conseguir usar a NN Yolo treinada)
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# 4. Iniciar a recepção/captura do Streaming

# 4.1 Recebe pacotes via protocolo RTMP - Real Time
# vs = VideoStream(streaming_path).start()

# 4.2 Recebe pacotes da camera local da máquina
# vs = VideoStream(src=0).start()

# 4.3 Recebendo pacotes a partir de um arquivo de vídeo carregado localmente
# PS.: não usar a função "VideoCapture" quando se quer "Real time". Essa função é frame to frame
vs = FileVideoStream(streaming_path).start()

# 4.4 Tempo de processamento da maquina local -> BOA PRÁTICA
time.sleep(1)

# 4.5 Inicia a captura dos Frames per Second
fps = FPS().start()
print("[+] Iniciando a leitura frame-frame do vídeo...")

# 5. PipeLine para iterar sobre os frames do streaming
while True:
    # 5.1 recepção do vídeo, e convertendo em um objeto do tipo array
    frame = vs.read()

    # # 5.2 Redimensionar os frames: cada NN espera receber inputs de tamanhos limites máximos (vide documentação de cada NN)
    # frame = cv2.resize(frame, None, fx=0.4, fy=0.4)

    # 5.3 Extraindo Altura e Largura do frame original capturado (importante para devolver a normalização)
    (H, W) = frame.shape[:2]

    # 5.4 Normalizando os frames (container blob), para auxiliar o uso da NN
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    # 5.5 Usando a rede NN no frame normalizado fazendo uma passagem (forward) na YOLO
    net.setInput(blob)

    # 5.6 Detectando os objetos
    layer_outputs = net.forward(ln)

    # 5.7 criar listas com boxes, nível de confiança e ids das classes, para iterar
    boxes = []
    confidences = []
    class_ids = []

    # 5.8 Iterações ao longo das detecções
    for output in layer_outputs:
        for detection in output:
            # 5.8.1 extrair níveis de confiança de detecção dos objetos detetctáveis
            scores = detection[5:]
                # 5.8.1.1 Classificar os scores
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # 5.8.2 extrair as coordenadas que possuem nivel de confiança maior que "treshold" passado como variavel global
            if confidence > CONFIDENCE_MIN and class_id in [0, 1, 2, 3]:
                box = detection[0:4] * np.array([W, H, W, H])
                (center_x, center_y, width, height) = box.astype("int")

                x = int(center_x - (width / 2))
                y = int(center_y - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # 5.9 Eliminar ruído e redundâncias aplicando "non-maxima suppression"
    new_ids = cv2.dnn.NMSBoxes(boxes, confidences,CONFIDENCE_MIN, NMS_THRESHOLD)
    if len(new_ids) > 0:
        for i in new_ids.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # 5.10 Plotar "retângulo" e "texto" das classes detectadas no frame atual
            color_picked = [int(c) for c in colors[class_ids[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color_picked, 2)
            text = "{}: {:.4f}".format(labels[class_ids[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_picked, 2)

    # 6. Exibir o frame atual
    cv2.imshow('frame', frame)

    # 6.1 Tecla Fuga "tecla ESC"
    c = cv2.waitKey(1)
    if c == 27:
        break

    # atualiza o fps
    fps.update()

# 7. Eliminar processos e janelas
fps.stop()
cv2.destroyAllWindows()
vs.stop()