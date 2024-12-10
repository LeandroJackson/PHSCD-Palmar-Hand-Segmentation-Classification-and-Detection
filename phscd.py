"""
PHSCD: Palmar Hand Segmentation, Classification, and Detection

Versão: 1.0.0

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

def cropHand(hand, model_path="model/detectHand.pt", conf_threshold=0.4, padding=0):
    
    """
    Recorta a região da mão na imagem utilizando um modelo YOLO para detecção.

    Params:
    - hand (np.ndarray): A imagem de entrada contendo a mão a ser detectada.
    - model_path (str): Caminho para o modelo YOLO treinado para detectar a mão.
    - conf_threshold (float): Limiar de confiança para considerar uma detecção válida.
    - padding (int): Quantidade de padding a ser adicionada ao redor da bounding box da mão.

    Returns:
    - np.ndarray: A imagem recortada contendo apenas a mão, com padding opcional.
    """

    model_detect = YOLO(model_path)
    
    if len(hand.shape) == 2:
        hand = cv2.merge([hand, hand, hand])

    results = model_detect.predict(source=hand, conf=conf_threshold)

    cropped_img = None

    for result in results:
        boxes = result.boxes
        names = result.names

        for box in boxes:

            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])

            cropped_img = hand[y_min:y_max, x_min:x_max]

            height, width = cropped_img.shape[:2]

            if height > width:                 # Se a altura for maior que a largura, adicionar bordas horizontais

                bordas = height - width
                left_border = bordas // 2
                right_border = bordas - left_border
                cropped_img = cv2.copyMakeBorder(cropped_img, 0, 0, left_border, right_border, cv2.BORDER_CONSTANT, value=(255, 255, 255))

            elif width > height:    # Se a largura for maior que a altura, adicionar bordas verticais

                bordas = width - height
                top_border = bordas // 2
                bottom_border = bordas - top_border
                cropped_img = cv2.copyMakeBorder(cropped_img, top_border, bottom_border, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))

            height, width = cropped_img.shape[:2]
            if height != width:
                min_side = min(height, width)
                cropped_img = cv2.resize(cropped_img, (min_side, min_side))

            class_idx = int(box.cls[0])
            class_name = names[class_idx]
            break

    return cropped_img



def segmentation(img, model_path = "yolov8m-seg.pt", confidence: float = 0.6) -> np.ndarray:
    """
    Função que retorna a máscara binária da segmentação de "person" de uma imagem usando um modelo YOLO.

    Args:
    - image_path (str): Caminho para a imagem de entrada.
    - model_path (str): Caminho para o modelo YOLO pré-treinado.
    - confidence (float): Limite de confiança para a detecção.

    Returns:
    - np.ndarray: A máscara binária da segmentação.
    """
    model = YOLO(model_path)

    results = model.predict(source=img, conf=confidence)

    for result in results:
        img = np.copy(result.orig_img)

        for ci, c in enumerate(result):
            label = c.names[c.boxes.cls.tolist().pop()]

            if label == "person":
                b_mask = np.zeros(img.shape[:2], np.uint8)

                contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
                cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)

                return b_mask

    return None
    
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


def maskHand(image):
    """
    Processa a imagem realizando recorte, segmentação, preenchimento de buracos, 
    dilatação da máscara e aplicação da máscara na imagem.

    Params:
    - image (np.ndarray): A imagem de entrada a ser processada.

    Returns:
    - np.ndarray: A imagem final após o processamento com a máscara aplicada.
    """

    crop_img = cropHand(image)

    mascara = segmentation(crop_img)

    if mascara is None or mascara.size == 0:
        return None

    
    buraco = fill_small_holes(mascara, 100)

    kernel = np.ones((5, 5), np.uint8)
    mascara_dilatada = cv2.dilate(buraco, kernel, iterations=5)

    image_result = cv2.bitwise_and(crop_img, crop_img, mask=mascara_dilatada)

    return image_result



def folder_process(folder_path, mode):
    """
    Processa todas as imagens JPEG em uma pasta, aplicando segmentação ou recorte dependendo do modo especificado.

    Params:
    - folder_path (str): Caminho para a pasta contendo as imagens a serem processadas.
    - mode (str): Modo de operação ('S' para segmentação, 'C' para recorte).

    Returns:
    - bool: Retorna True se o processamento for bem-sucedido.
    """

    backup_filename = os.path.basename(folder_path) + '_BACKUP.zip'
    
    with zipfile.ZipFile(backup_filename, 'w', zipfile.ZIP_DEFLATED) as backup_zip:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                backup_zip.write(file_path, os.path.relpath(file_path, folder_path))
    
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.jpg'):
                file_path = os.path.join(root, file)
                img = cv2.imread(file_path)
                
                if mode == 'S':
                    processed_img = maskHand(img)
                elif mode == 'C':
                    processed_img = cropHand(img)
                else:
                    print("Modo não especificado, C -> Crop (Recorte da mão) S - > segmentation (Segmentação da mão)")
                    return
                
                if processed_img is None or processed_img.size == 0:
                    print("Mão não encontrada, arquivo: ", file_path)
                    continue

                cv2.imwrite(file_path, processed_img)

    return True