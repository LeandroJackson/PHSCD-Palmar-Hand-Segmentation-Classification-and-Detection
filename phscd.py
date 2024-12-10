import cv2
import numpy as np
from ultralytics import YOLO
import zipfile
import os

#### Palmar Hand Segmentation, Classification, and Detection

# Função para adicionar padding à bounding box
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
    Recorta a região da mão da imagem usando um modelo YOLO.

    :param hand: A imagem de entrada contendo a mão.
    :param model_path: O caminho para o modelo YOLO.
    :param conf_threshold: O limiar de confiança para a detecção.
    :param padding: A quantidade de padding a ser adicionada ao redor da bounding box.
    :return: A imagem recortada contendo apenas a mão.
    """
    model_detect = YOLO(model_path)
    
    if len(hand.shape) == 2:
        hand = cv2.merge([hand, hand, hand])

    img_height, img_width = hand.shape[:2]

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
    Preenche buracos pequenos em uma máscara com base no tamanho do contorno.

    :param img_thresh: A máscara binária onde os buracos precisam ser preenchidos.
    :param threshold: O tamanho mínimo para considerar um contorno como "pequeno" e preenchê-lo.
    :return: A máscara com buracos pequenos preenchidos.
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
    Função que processa a imagem, realizando o recorte, segmentação, preenchimento de buracos,
    dilatação da máscara e aplicando a máscara na imagem.

    :param image_path: O caminho para a imagem de entrada.
    :return: A imagem final após o processamento com a máscara aplicada.
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