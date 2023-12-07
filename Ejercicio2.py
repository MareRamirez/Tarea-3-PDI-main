#Margareth Ramirez Valenzuela, Ana María Vargas
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 2.1 Aplique a la imagen un filtro que le permita obtener el gradiente optimo para la aplicacion
# del algoritmo Level Set y aplıquelo a la imagen original. Explique por que dicho filtro inicial
# es necesario para el funcionamiento del algoritmo de Level Set.

# Se carga la imagen
imagen = 'RM_T1.png'
img = cv2.imread(imagen, 0)  # Se transforma a escala de grises

# Se aplica el filtro Gaussiano para reducir el ruido y mejorar la detección de bordes
blurred_image = cv2.GaussianBlur(img, (5, 5), 0)

# Se aplica el filtro Sobel 
sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=5)
sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=5)
magnitud = cv2.magnitude(sobel_x, sobel_y)

# Se normaliza la imagen de gradiente al rango [0,255]
magnitud = cv2.normalize(magnitud, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Se invierten los colores de la imagen para asemejarla a la imagen de ejemplo
inverted_gradient = cv2.bitwise_not(magnitud)

# Se muestra la imagen original junto a la imagen 
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(inverted_gradient, cmap='gray')
plt.title('Inverted Gradient Image')
plt.axis('off')

plt.show()

# En este caso se ha escogido el filtro Sobel ya que este filtro es excelente para la detección de bordes. Este se aplica 
# horizontal y verticalmente logrando una detección más detallada para después poder aplicar el algoritmo de Level Set.
