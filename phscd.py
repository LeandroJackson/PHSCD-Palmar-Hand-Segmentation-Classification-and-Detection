"""
PHSCD: Palmar Hand Segmentation, Classification, and Detection

Versão: 2.0.0

Autor: Jackson Leandro
Email: jacksonnascimento@eng.ci.ufpb.br

Descrição:
Este código foi desenvolvido para detectar, segmentar, classificar e recortar mãos em imagens,
com foco na biometria palmar. Utiliza o modelo YOLO (You Only Look Once), que foi treinado com 13 mil imagens de mãos.

Créditos:
Este código foi desenvolvido integralmente por Jackson Leandro para fins acadêmicos e de pesquisa. 
Qualquer uso, modificação ou distribuição deve ser devidamente creditado ao autor original. 
Para questões ou colaborações, entre em contato através do email fornecido.

Nota:
Ao longo do tempo, planejo aprimorar continuamente o modelo de clasificação e de segmentação, 
que atualmente necessita de mais treinamento e refinamento para alcançar uma maior 
precisão e robustez na identificação e análise das mãos.

Dependências:
- OpenCV
- NumPy
- Ultralytics YOLO
"""


import cv2
import numpy as np
from ultralytics import YOLO
import zipfile
import os



def add_padding(x_min, y_min, x_max, y_max, padding, img_width, img_height):
    """
    Adiciona padding às coordenadas da bounding box, garantindo que os valores permaneçam dentro dos limites da imagem.
    """

    x_min_padded = max(0, x_min - padding)
    y_min_padded = max(0, y_min - padding)
    x_max_padded = min(img_width, x_max + padding)
    y_max_padded = min(img_height, y_max + padding)
    return x_min_padded, y_min_padded, x_max_padded, y_max_padded

def coordinateHand(hand, model, conf_threshold=0.4, padding=0):
    
    """
    Detecta a coordenada da mão na imagem utilizando um modelo YOLO para detecção.

    Params:
    - hand (np.ndarray): A imagem de entrada contendo a mão a ser detectada.
    - model_path (str): Caminho para o modelo YOLO treinado para detectar a mão.
    - conf_threshold (float): Limiar de confiança para considerar uma detecção válida.
    - padding (int): Quantidade de padding a ser adicionada ao redor da bounding box da mão.

    Returns:
    - np.ndarray: Retorna a lista de coordenada.
    """
    
    if len(hand.shape) == 2:
        hand = cv2.merge([hand, hand, hand])

    results = model.predict(source=hand, conf=conf_threshold, device="cpu", half=False, verbose=False, imgsz=480)

    if not results or len(results[0].boxes) == 0:
        return None


    for result in results:
        boxes = result.boxes
        names = result.names

        for box in boxes:

            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
            coordinates = (x_min, y_min, x_max, y_max)

            class_idx = int(box.cls[0])
            break

    return coordinates

def cropHand(imagem, coordinates):

    """
    Função que recorta uma região da imagem com base nas coordenadas fornecidas
    e ajusta as bordas da imagem para garantir que ela tenha uma proporção quadrada.

    Params:
    - imagem: A imagem de entrada (tipo numpy.ndarray).
    - coordinates: As coordenadas (x_min, y_min, x_max, y_max) para definir a área de recorte.
    Returns:
    - A imagem recortada e ajustada, com proporções quadradas.
    """

    x_min, y_min, x_max, y_max = coordinates

    cropped_img = imagem[y_min:y_max, x_min:x_max]
    #cropped_img2 = cropped_img

    height, width = cropped_img.shape[:2]

    if height > width:              

        bordas = height - width
        left_border = bordas // 2
        right_border = bordas - left_border
        cropped_img = cv2.copyMakeBorder(cropped_img, 0, 0, left_border, right_border, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    elif width > height:  

        bordas = width - height
        top_border = bordas // 2
        bottom_border = bordas - top_border
        cropped_img = cv2.copyMakeBorder(cropped_img, top_border, bottom_border, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    height, width = cropped_img.shape[:2]
    if height != width:
        min_side = min(height, width)
        cropped_img = cv2.resize(cropped_img, (min_side, min_side))

    return cropped_img


def detect_hand_info(img: np.ndarray, model, conf_threshold=0.5, padding=0) -> tuple:
    """
    Detecta a mão em uma imagem utilizando YOLO e retorna as coordenadas da bounding box e a máscara binária de segmentação.

    Params:
    - img (np.ndarray): Imagem de entrada.
    - model_path (str): Caminho para o modelo YOLO treinado.
    - conf_threshold (float): Limiar de confiança da detecção.
    - padding (int): Padding extra para aplicar na bounding box.

    Returns:
    - Tuple: (coordenadas da bounding box (x_min, y_min, x_max, y_max), máscara binária np.ndarray)
             Retorna (None, None) se nenhuma detecção válida for encontrada.
    """

    if len(img.shape) == 2:
        img = cv2.merge([img, img, img])

    #results = model.predict(source=img, conf=conf_threshold)
    results = model.predict(source=img, conf=conf_threshold, device="cpu", half=False, verbose=False, imgsz=480)


    if not results or len(results[0].boxes) == 0:
        return None, None

    for result in results:
        img_copy = np.copy(result.orig_img)
        names = result.names

        for i, box in enumerate(result.boxes):
            class_idx = int(box.cls[0])
            label = names[class_idx]

            if label != "hand":
                continue

            # Coordenadas da bounding box com padding opcional
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
            x_min = max(x_min - padding, 0)
            y_min = max(y_min - padding, 0)
            x_max = min(x_max + padding, img.shape[1])
            y_max = min(y_max + padding, img.shape[0])
            coordinates = (x_min, y_min, x_max, y_max)

            # Máscara binária da segmentação
            if result.masks is not None and result.masks.xy:
                contour = result.masks.xy[i].astype(np.int32).reshape(-1, 1, 2)
                b_mask = np.zeros(img.shape[:2], dtype=np.uint8)
                cv2.drawContours(b_mask, [contour], -1, 255, cv2.FILLED)
            else:
                b_mask = None

            return coordinates, b_mask

    return None, None

    
def fill_small_holes(img_thresh, threshold=10000):
    """
    Preenche buracos pequenos em uma máscara binária com base no tamanho do contorno.

    Params:
    - img_thresh (np.ndarray): A máscara binária onde os buracos precisam ser preenchidos.
    - threshold (int): O tamanho mínimo para considerar um contorno como "pequeno" e preenchê-lo.

    Returns:
    - np.ndarray: A máscara com buracos pequenos preenchidos.
    """
    contours, _ = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    small_holes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < threshold:
            small_holes.append(contour)

    cv2.drawContours(img_thresh, small_holes, -1, 255, -1)

    return img_thresh


def maskHand(image, model):
    """
    Processa a imagem realizando a identificação da mão, segmentação, preenchimento de buracos, 
    dilatação da máscara e aplicação da máscara na imagem.

    Params:
    - image (np.ndarray): A imagem de entrada a ser processada.

    Returns:
    - np.ndarray: A imagem final após o processamento com a máscara aplicada.
    - list: coodernadas do bounding box
    - np.ndarray: A mascara da imagem completa
    """

    coordinates, mascara = detect_hand_info(image, model)

    if coordinates is None:
        return None, None, None

    
    buraco = fill_small_holes(mascara, 100)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_smooth1 = cv2.morphologyEx(buraco, cv2.MORPH_CLOSE, kernel, iterations=2)

    mask_smooth2 = cv2.medianBlur(mask_smooth1, 29)



    kernel = np.ones((1, 1), np.uint8)
    image_mask_seg = cv2.dilate(mask_smooth2, kernel, iterations=3)

    image_result = cv2.bitwise_and(image, image, mask=image_mask_seg)

    crop_hand_seg = cropHand(image_result, coordinates)

    return crop_hand_seg, coordinates, image_mask_seg 


def desenhar_bounding_box(img, bbox, espessura=20, cor=(0, 255, 0), ):
    x1, y1, x2, y2 = bbox
    return cv2.rectangle(img.copy(), (x1, y1), (x2, y2), cor, espessura)


def linha_dentro_mascara(p1, p2, masc_gray, num_pontos=5):
    """Verifica se a linha entre p1 e p2 está inteiramente dentro da máscara."""
    for i in range(num_pontos):
        t = i / (num_pontos - 1) if num_pontos > 1 else 0.5
        x = int(round(p1[0] * (1 - t) + p2[0] * t))
        y = int(round(p1[1] * (1 - t) + p2[1] * t))
        if x < 0 or x >= masc_gray.shape[1] or y < 0 or y >= masc_gray.shape[0]:
            return False
        if masc_gray[y, x] == 0:
            return False
    return True

def desenhar_triangulos_biometria(img, mascara, densidade_pontos=2000, espessura=2, alpha=1.0, color = (0, 255, 0)):
    masc_gray = cv2.cvtColor(mascara, cv2.COLOR_BGR2GRAY) if len(mascara.shape) == 3 else mascara.copy()
    contornos, _ = cv2.findContours(masc_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return img
    
    overlay = img.copy()
    altura, largura = masc_gray.shape

    # Coleta pontos dos contornos
    pontos_contorno = []
    for cnt in contornos:
        if len(cnt) > 0:
            cnt = cnt.squeeze()
            if cnt.ndim == 1:
                pontos_contorno.append(cnt)
            else:
                for p in cnt:
                    pontos_contorno.append(p)
    pontos = np.array(pontos_contorno, dtype=np.int32) if pontos_contorno else np.empty((0, 2), dtype=np.int32)

    # Adiciona pontos aleatórios dentro da máscara
    for _ in range(densidade_pontos):
        x = np.random.randint(0, largura)
        y = np.random.randint(0, altura)
        if masc_gray[y, x] > 0:
            pontos = np.append(pontos, [[x, y]], axis=0)

    # Triangulação de Delaunay
    if pontos.size == 0:
        return img
    subdiv = cv2.Subdiv2D(cv2.boundingRect(pontos))
    for p in pontos:
        subdiv.insert((int(p[0]), int(p[1])))
    triangulos = subdiv.getTriangleList().astype(np.int32)

    # Desenha triângulos válidos
    for t in triangulos:
        pts = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
        
        vertices_validos = all(0 <= pt[0] < largura and 0 <= pt[1] < altura and masc_gray[pt[1], pt[0]] > 0 for pt in pts)
        if not vertices_validos:
            continue
        
        arestas_validas = True
        for i in range(3):
            p1 = pts[i]
            p2 = pts[(i+1) % 3]
            if not linha_dentro_mascara(p1, p2, masc_gray, num_pontos=5):
                arestas_validas = False
                break
        if not arestas_validas:
            continue
        
        cv2.polylines(overlay, [np.array(pts)], isClosed=True, color=color, thickness=espessura)

    img_final = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    return img_final


def folder_process(model, folder_path, mode):
    """
    Processa todas as imagens JPEG em uma pasta, aplicando segmentação ou recorte dependendo do modo especificado.

    Params:
    - folder_path (str): Caminho para a pasta contendo as imagens a serem processadas.
    - mode (str): Modo de operação ('S' para segmentação, 'C' para recorte, 'M' para as mascaras binarias).

    Returns:
    - bool: Retorna True se o processamento for bem-sucedido.
    """
    
    backup_filename = os.path.basename(folder_path) + '_BACKUP.zip'
    caminhos_sem_mao = []

    
    with zipfile.ZipFile(backup_filename, 'w', zipfile.ZIP_DEFLATED) as backup_zip:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                backup_zip.write(file_path, os.path.relpath(file_path, folder_path))
    
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(root, file)
                img = cv2.imread(file_path)

                hand_seg, coordinates, mask = maskHand(img, model)
                
                if mode == 'S':
                    processed_img = hand_seg
                elif mode == 'C':
                    processed_img = cropHand(img, coordinates)
                elif mode == 'M':
                    processed_img = mask

                else:
                    print("Modo não especificado, C -> Crop (Recorte da mão) S - > segmentation (Segmentação da mão)")
                    return
                
                if processed_img is None or processed_img.size == 0:
                    print("Mão não encontrada, arquivo: ", file_path)
                    caminhos_sem_mao.append(file_path) 
                    continue

                cv2.imwrite(file_path, processed_img)
    
    if caminhos_sem_mao:
        with open("imagens_sem_mao.txt", "w") as f:
            for caminho in caminhos_sem_mao:
                f.write(caminho + "\n")

    return True