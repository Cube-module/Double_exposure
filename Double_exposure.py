import cv2 #BGR
import numpy as np
import requests
import sys
import io
from PIL import Image as im #RGB
import matplotlib.pyplot as plt # RGB

# Импортируем специальную функцию для Colab (если работаете в Google Colab)
from google.colab.patches import cv2_imshow



url1 = "https://wallpapers.com/images/hd/iori-kitahara-standing-on-the-ledge-of-the-ocean-o1vsmby83sz5y57z.jpg"
url2= "https://encrypted-tbn2.gstatic.com/images?q=tbn:ANd9GcQLVI5oqw867VkKFwDFtPaTGx2iRyLI2xth6m96yXdszR9vcfGi"

# отправляем get запрос
response_url1 = requests.get(url1)
response_url2 = requests.get(url2)

# проверяем что при get запросе не произошла ошибка
a = response_url1.status_code
b = response_url2.status_code

if a != 200:
  print("Ошибка загрузки первого изображения", a)
  sys.exit(1)

if b != 200:
  print("Ошибка загрузки второго изображения", b)
  sys.exit(1)

# получаем байты изображения
bytes_image1 = response_url1.content
bytes_image2 = response_url2.content

# создаем виртуальный файл (лежит на оперативке)
buffer_image1 = io.BytesIO(bytes_image1)
buffer_image2 = io.BytesIO(bytes_image2)

# получаем полноценное изображение PIL (RGB)
pil_image1 = im.open(buffer_image1)
pil_image2 = im.open(buffer_image2)

# преобразуем изображение PIL в матрицу (NumPy arrays)
square_im1 = np.array(pil_image1)
square_im2 = np.array(pil_image2)

# переводим цвета из RGB в BGR для cv2
# cv2.COLOR_RGB2BGR константа для преобразования преобразования (4)
bgr_image1 = cv2.cvtColor(square_im1, cv2.COLOR_RGB2BGR)
bgr_image2 = cv2.cvtColor(square_im2, cv2.COLOR_RGB2BGR)

# подгоняем размеры изображений друг под друга
resize_im1 = cv2.resize(bgr_image1, (700, 500))
resize_im2 = cv2.resize(bgr_image2, (700, 500))

# устанавливаем веса для смешивания 
alpha = 0.7 # im1
beta = 0.3  # im2
gamma = 0   # яркость

# смешиваем картинки (изменяем интенсивность свечения диодов в пикселе)
new_pixels = cv2.addWeighted(resize_im1, alpha, resize_im2, beta, gamma)



# cv2_imshow(new_pixels)

# задаем соотношение сторон
plt.figure(figsize=(30, 10))

# размещаем 3 картинки для наглядности
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(resize_im1, cv2.COLOR_BGR2RGB))
plt.title('Image 1, 0.7')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(resize_im2, cv2.COLOR_BGR2RGB))
plt.title('Image 2, 0.3')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(new_pixels, cv2.COLOR_BGR2RGB))
plt.title('Blended Image, 0.7/0.3')
plt.axis('off')

plt.show()
