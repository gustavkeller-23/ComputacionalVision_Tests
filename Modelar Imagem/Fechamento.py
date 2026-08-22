"""
Script de Morfologia Matemática em Imagens
--------------------------------------------
Recebe uma imagem, converte para preto e branco (escala de cinza / binária)
e aplica as operações morfológicas: Erosão, Dilatação, Abertura e Fechamento.

Requisitos:
    pip install opencv-python numpy matplotlib
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ------------------------------------------------------
# 1. CONFIGURAÇÕES
# ------------------------------------------------------
CAMINHO_IMAGEM = "FONE.jpg"      # <-- coloque aqui o caminho da sua imagem
PASTA_SAIDA = "resultados"          # pasta onde as imagens processadas serão salvas
KERNEL_SIZE = 5                     # tamanho do "elemento estruturante" (ajuste conforme necessário)
ITERACOES = 1                       # quantas vezes aplicar erosão/dilatação


def main():
    # Cria a pasta de saída, se não existir
    os.makedirs(PASTA_SAIDA, exist_ok=True)

    # ------------------------------------------------------
    # 2. CARREGAR A IMAGEM
    # ------------------------------------------------------
    imagem_original = cv2.imread(CAMINHO_IMAGEM)

    if imagem_original is None:
        raise FileNotFoundError(
            f"Não foi possível carregar a imagem em '{CAMINHO_IMAGEM}'. "
            "Verifique o caminho/nome do arquivo."
        )

    # ------------------------------------------------------
    # 3. CONVERTER PARA PRETO E BRANCO
    # ------------------------------------------------------
    # 3a. Escala de cinza (tons de cinza, não apenas 0/255)
    imagem_cinza = cv2.cvtColor(imagem_original, cv2.COLOR_BGR2GRAY)

    # 3b. Binarização (preto e branco "puro": 0 ou 255)
    # Usamos Otsu para encontrar automaticamente o melhor limiar (threshold)
    _, imagem_binaria = cv2.threshold(
        imagem_cinza, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # ------------------------------------------------------
    # 4. ELEMENTO ESTRUTURANTE (kernel)
    # ------------------------------------------------------
    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)

    # ------------------------------------------------------
    # 5. OPERAÇÕES MORFOLÓGICAS
    # ------------------------------------------------------
    # Erosão: "encolhe" as áreas brancas (remove pequenos detalhes/ruídos)
    erosao = cv2.erode(imagem_binaria, kernel, iterations=ITERACOES)

    # Dilatação: "expande" as áreas brancas (engrossa formas, fecha buracos pequenos)
    dilatacao = cv2.dilate(imagem_binaria, kernel, iterations=ITERACOES)

    # Abertura: Erosão seguida de Dilatação (remove ruídos pequenos mantendo o tamanho geral)
    abertura = cv2.morphologyEx(imagem_binaria, cv2.MORPH_OPEN, kernel)

    # Fechamento: Dilatação seguida de Erosão (fecha pequenos buracos/falhas nos objetos)
    fechamento = cv2.morphologyEx(imagem_binaria, cv2.MORPH_CLOSE, kernel)

    # ------------------------------------------------------
    # 6. SALVAR OS RESULTADOS
    # ------------------------------------------------------
    cv2.imwrite(os.path.join(PASTA_SAIDA, "1_cinza.png"), imagem_cinza)
    cv2.imwrite(os.path.join(PASTA_SAIDA, "2_binaria.png"), imagem_binaria)
    cv2.imwrite(os.path.join(PASTA_SAIDA, "3_erosao.png"), erosao)
    cv2.imwrite(os.path.join(PASTA_SAIDA, "4_dilatacao.png"), dilatacao)
    cv2.imwrite(os.path.join(PASTA_SAIDA, "5_abertura.png"), abertura)
    cv2.imwrite(os.path.join(PASTA_SAIDA, "6_fechamento.png"), fechamento)

    print(f"Imagens salvas na pasta '{PASTA_SAIDA}':")
    print("  1_cinza.png       -> escala de cinza")
    print("  2_binaria.png     -> preto e branco (binarizada)")
    print("  3_erosao.png      -> erosão")
    print("  4_dilatacao.png   -> dilatação")
    print("  5_abertura.png    -> abertura")
    print("  6_fechamento.png  -> fechamento")

    # ------------------------------------------------------
    # 7. MOSTRAR TUDO EM UMA ÚNICA JANELA (opcional)
    # ------------------------------------------------------
    imagens = [
        ("Original (cinza)", imagem_cinza),
        ("Binária (P&B)", imagem_binaria),
        ("Erosão", erosao),
        ("Dilatação", dilatacao),
        ("Abertura", abertura),
        ("Fechamento", fechamento),
    ]

    plt.figure(figsize=(12, 8))
    for i, (titulo, img) in enumerate(imagens, start=1):
        plt.subplot(2, 3, i)
        plt.imshow(img, cmap="gray")
        plt.title(titulo)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_SAIDA, "0_comparativo.png"))
    plt.show()


if __name__ == "__main__":
    main()