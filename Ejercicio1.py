#Margareth Ramirez Valenzuela, Ana María Vargas
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters

# Parte 1.A
# Cargue y despliegue la imagen en escala de grises
img = cv2.imread('moon.png',0)
plt.imshow(img, cmap='gray')
plt.title('Imagen original')
plt.axis('off')
plt.show()

# Parte 1.B
# Transformada de Fourier
img_fft = np.fft.fft2(img)
# Centrar el espectro de frecuencia
img_fft_shift = np.fft.fftshift(img_fft)
# Espectro de amplitud
img_fft_abs = np.abs(img_fft_shift)
# Espectro de amplitud en escala logarítmica
img_fft_log = np.log(img_fft_abs)

# Despliegue del espectro de frecuencia centrado
plt.imshow(img_fft_log, cmap='gray')
plt.title('Espectro de frecuencia centrado')
plt.axis('off')
plt.show()

# Parte 1.C
# Para lograr eliminar el ruido en la imagen ocasionado por las frecuencias especıficas, primero
# obtenga una mascara a partir del modulo de la Transformada de Fourier en escala logar ́ıtmica
# utilizando un umbral igual a 10. Para unir los puntos mas cercanos aplique un filtro de
# mınimo.

# Umbral
umbral = 10
# Mascara
mask = img_fft_log > umbral
# Filtro de mínimo
mask = filters.minimum_filter(mask, size=3)




