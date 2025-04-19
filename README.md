# PHSCD - Palmar Hand Segmentation, Classification, and Detection

PHSCD é uma tecnologia desenvolvida para detectar, segmentar, classificar e recortar mãos em imagens, com aplicação na biometria palmar. Utilizando o modelo YOLO (You Only Look Once), que foi treinado com 13 mil imagens de mãos, o código realiza diversas operações de processamento de imagem para isolar e manipular regiões de interesse contendo mãos.

Este `README` demonstra o funcionamento das principais funções do módulo `phscd.py`, utilizado para segmentação, detecção e recorte de mãos em imagens, com aplicação em biometria palmar.  



### **Estrutura do Projeto**

Abaixo está a estrutura de diretórios e arquivos do projeto:

```plaintext
.
├── README.md             # Documentação do projeto
├── requirements.txt      # Dependências necessárias para o projeto
├── phscd.py              # Script principal para segmentação e detecção de mãos
├── model                 # Diretório contendo modelos treinados
│   ├── detectHand.pt     # Modelo treinado para detecção de mãos
```



### **Instalação**

1. **Clone o repositório:**

   Primeiro, clone o repositório usando o comando abaixo:

   ```bash
   git clone https://github.com/LeandroJackson/PHSCD-Palmar-Hand-Segmentation-Classification-and-Detection.git
   ```

2. **Crie um ambiente virtual:**

   Crie um ambiente Conda ou um ambiente virtual Python.

   ```bash
   conda create --name biometria python=3.10
   ```

   Ambiente virtual com `venv`:

   ```bash
   python -m venv biometria
   ```

3. **Ative o ambiente:**

   Para ativar o ambiente Conda, use:

   ```bash
   conda activate biometria
   ```

   Caso tenha usado `venv`, ative o ambiente com:

   ```bash
   source biometria/bin/activate   # No Linux ou MacOS
   biometria\Scripts\activate      # No Windows
   ```

4. **Instale as dependências:**

   Instale o pacote `ultralytics`::

   ```bash
   pip install ultralytics
   ```




```python
from phscd import phscd
import matplotlib.pyplot as plt
import cv2
```

### O modelo treinado utilizado está localizado na pasta `model`, com o nome `yolo_hand_phscd.pt`


```python
model = phscd.YOLO("phscd/model/yolo_hand_phscd.pt")
```

#### Exemplo de Uso

Imagem pública obtida da base de dados CASIA. Disponível em: [http://english.ia.cas.cn/db/201611/t20161101_169937.html](http://english.ia.cas.cn/db/201611/t20161101_169937.html)


```python
img_bgr = cv2.imread('exemplo.jpg')
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
```

### Exemplo de Uso - Segmentação da Mão

A função `phscd.maskHand(imagem, model)`realiza a segmentação da mão presente na imagem.  


**Retorno:**  
- Imagem da mão recortada com dimensões quadradas (largura e altura iguais), pronta para uso em modelos de predição, como redes siamesas;  
- Coordenadas da bounding box, referentes à resolução da imagem original;  
- Máscara binária da segmentação, gerada com base na resolução da imagem original.



```python
imagem_segmentada, coords, mascara = phscd.maskHand(img_gray, model)
```


```python
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

titulos = ["Imagem Original", "Imagem Segmentada", "Máscara Binária"]
imagens = [img_gray, imagem_segmentada, mascara]

for ax, img, titulo in zip(axs, imagens, titulos):
    ax.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
    ax.set_title(titulo)
    ax.axis('off')

plt.tight_layout()
plt.show()

```


    
![png](README_files/README_8_0.png)
    


### Exemplo de Uso - Recorte da Mão


A função `phscd.cropHand(imagem, coordenadas)` permite recortar o objeto de interesse, inserindo bordas pretas para manter a imagem com formato quadrado. Esse procedimento assegura que todas as mãos sejam mantidas na mesma proporção, evitando distorções morfológicas caso as imagens sejam redimensionadas.


```python
imagem_crop = phscd.cropHand(img_gray, coords)
plt.imshow(imagem_crop)
plt.axis('off')
plt.show()
```


    
![png](README_files/README_10_0.png)
    


### Exemplo de Uso - Desenho da Bounding Box na Mão

A partir das coordenadas retornadas pela função `phscd.maskHand`, é possível desenhar a bounding box diretamente sobre a imagem original.  


Mas para fins de otimização, é possível obter somente as coordenadas através da função `coordenadas = phscd.coordinateHand(imagem, modelo)`


```python
bounding_box = phscd.desenhar_bounding_box(img_gray, coords, 1)
plt.imshow(bounding_box)
plt.axis('off')
plt.show()
```


    
![png](README_files/README_12_0.png)
    


### Exemplo de Uso - Máscara com Triangulação Biométrica

A função `desenhar_triangulos_biometria(img, mascara, densidade_pontos=2000, espessura=2, alpha=1.0, color=(0, 255, 0))` realiza a representação visual de triângulos sobre a mão, com o objetivo de analisar o comportamento das máscaras em sistemas com tela LCD.


```python
triangulos = phscd.desenhar_triangulos_biometria(img_gray, mascara, 1024, 1)
plt.imshow(triangulos)
plt.axis('off')
plt.show()
```


    
![png](README_files/README_14_0.png)
    


### Exemplo de Uso - Processamento em Lote

A função `folder_process(model, folder_path, mode)` percorre recursivamente todas as pastas e subpastas dentro de `folder_path`, processando imagens no formato JPG, JPEG e PNG conforme o modo especificado:

- `'S'`: aplica segmentação da mão  
- `'C'`: realiza o recorte da mão  
- `'M'`: gera a máscara binária da mão  

Antes de qualquer modificação, a função gera um arquivo `.zip` como backup de todo o conteúdo original. Após isso, substitui cada imagem pela versão processada correspondente. Imagens em que a mão não é detectada são registradas no arquivo `imagens_sem_mao.txt`.

Retorna `True` se todo o processamento for concluído.
