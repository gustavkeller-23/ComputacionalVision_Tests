"""
Requisitos:
    pip install opencv-python numpy matplotlib
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


CAMINHO_IMAGEM = "FONE.jpg"         # caminho da sua imagem
PASTA_SAIDA = "resultados"          # pasta onde as imagens processadas serão salvas
KERNEL_SIZE = 5                     # tamanho do "elemento estruturante" (ajuste conforme necessário)
ITERACOES = 1                       # quantas vezes aplicar erosão/dilatação


def main():
    # Cria a pasta de saída, se não existir
    os.makedirs(PASTA_SAIDA, exist_ok=True)

    imagem_original = cv2.imread(CAMINHO_IMAGEM)

    if imagem_original is None:
        raise FileNotFoundError(
            f"Não foi possível carregar a imagem em '{CAMINHO_IMAGEM}'. "
            "Verifique o caminho/nome do arquivo."
        )

    # Escala de cinza (tons de cinza, não apenas 0/255)
    imagem_cinza = cv2.cvtColor(imagem_original, cv2.COLOR_BGR2GRAY)

    # Binarização (preto e branco "puro": 0 ou 255)
    # Usamos Otsu para encontrar automaticamente o melhor limiar (threshold)
    _, imagem_binaria = cv2.threshold(
        imagem_cinza, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)

    # Fechamento: Dilatação seguida de Erosão (fecha pequenos buracos/falhas nos objetos)
    fechamento = cv2.morphologyEx(imagem_binaria, cv2.MORPH_CLOSE, kernel)

    cv2.imwrite(os.path.join(PASTA_SAIDA, "fechamento.png"), fechamento)

    gerarImagem(fechamento, "Fechamento")

def gerarImagem(img, titulo):
    plt.figure(figsize=(12, 8))
    plt.imshow(img, cmap="gray")
    plt.title(titulo)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


main()