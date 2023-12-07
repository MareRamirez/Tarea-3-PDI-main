#Margareth Ramirez Valenzuela, Ana María Vargas
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pylab as pylab
from skimage import filters
from scipy.ndimage import filters

# Parte 1.A
# Cargue y despliegue la imagen en escala de grises
img = cv2.imread('moon.png',0)
plt.imshow(img, cmap=pylab.cm.Greys_r)
plt.title('Imagen original')
plt.axis('off')
plt.show()
 
print(img.shape)

# Parte 1.B
# Transformada de Fourier
img_fft = np.fft.fft2(img) # Fourier
img_fft_shift = np.fft.fftshift(img_fft) # Desplazamiento
img_fft_abs = np.abs(img_fft_shift) # Magnitud
img_fft_log = np.log(img_fft_abs) # Logaritmo
print(img_fft_log.shape)
plt.imshow(img_fft_log, cmap=pylab.cm.Greys_r)
plt.title('Espectro de frecuencia')
plt.axis('off')
plt.show()


# Parte 1.C
# Para lograr eliminar el ruido en la imagen ocasionado por las frecuencias especıficas, primero
# obtenga una mascara a partir del módulo de la Transformada de Fourier en escala logarıtmica
# utilizando un umbral igual a 10. Para unir los puntos más cercanos aplique un filtro de
# mınimo. 

# Umbral
umbral = 10
img_fft_log[img_fft_log < umbral] = 1
img_fft_log[img_fft_log >= umbral] = 0

# Filtro de mínimo
img_min = filters.minimum_filter(img_fft_log, size=3)

# Mostrar la máscara después del filtro de mínimo
plt.imshow(img_min, cmap='gray')
plt.title('Máscara Umbral')
plt.axis('off')
plt.show()


# Con la máscara del filtro selectivo suprima además las componentes de muy baja frecuencia.
# Para esto puede combinar la máscara obtenida en el paso anterior (c) con un filtro pasa bajo
# ideal de radio a elección (utilizar operaciones lógicas). Para la implementación del filtro ideal
# utilice la función disponible en el archivo fft lib. Despliegue las tres imágenes máscara, filtro
# ideal y combinación de ambos,

# Filtro pasa bajo ideal
def filtro_ideal(img, D0):
    # Dimensiones de la imagen
    M, N = img.shape
    # Coordenadas del centro
    u0 = M/2
    v0 = N/2
    # Filtro ideal
    filtro = np.zeros((M,N))
    for u in range(M):
        for v in range(N):
            D = np.sqrt((u-u0)**2 + (v-v0)**2)
            if D <= D0:
                filtro[u,v] = 1
    return filtro

# Filtro ideal
D0 = 10
filtro = filtro_ideal(img, D0)
plt.imshow(filtro, cmap='gray')
plt.title('Filtro ideal')
plt.axis('off')
plt.show()

# Combinación de ambos
img_min1 = img_min* filtro
plt.imshow(img_min1, cmap='gray')
plt.title('Combinación')
plt.axis('off')
plt.show()



