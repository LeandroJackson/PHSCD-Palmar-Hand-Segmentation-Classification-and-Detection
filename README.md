# PHSCD: Palmar Hand Segmentation, Classification, and Detection

PHSCD é uma tecnologia desenvolvida para detectar, segmentar, classificar e recortar mãos em imagens, com foco na biometria palmar. Utilizando o modelo YOLO (You Only Look Once), que foi treinado com 13 mil imagens de mãos, o código realiza diversas operações de processamento de imagem para isolar e manipular regiões de interesse contendo mãos.

## Funcionalidades

- **Detecção de Mãos**: Utiliza um modelo YOLO pré-treinado para detectar a presença de mãos em uma imagem.
- **Recorte de Mãos**: Recorta a região da mão detectada e adiciona bordas para normalizar as dimensões.
- **Segmentação de Mãos**: Gera uma máscara binária para a região da mão usando um modelo de segmentação.
- **Preenchimento de Buracos**: Preenche buracos pequenos na máscara de segmentação.
- **Processamento em Lote**: Processa todas as imagens em uma pasta, aplicando a detecção, segmentação e recorte de mãos conforme especificado.

## Estrutura do Projeto

```plaintext
.
├── README.md
├── requirements.txt
├── src
│   ├── detect_crop.py
│   ├── segmentation.py
│   ├── utils.py
│   └── main.py
├── models
│   ├── detectHand.pt
│   └── yolov8m-seg.pt
└── examples
    └── sample_image.jpg
