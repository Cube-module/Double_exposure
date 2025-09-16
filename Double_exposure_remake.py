import cv2
import numpy as np
import requests
import sys
import io
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, Optional

def download_image(url: str, timeout: int = 10) -> bytes:
    """
    Загружает изображение по URL
    
    Args:
        url: URL изображения
        timeout: таймаут запроса в секундах
        
    Returns:
        bytes: содержимое изображения в байтах
        
    Raises:
        requests.exceptions.RequestException: при ошибке загрузки
    """
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()  # Вызовет исключение для кодов 4xx/5xx
        return response.content
    except requests.exceptions.RequestException as e:
        raise Exception(f"Ошибка загрузки изображения по URL {url}: {e}")

def bytes_to_cv2_image(image_bytes: bytes) -> np.ndarray:
    """
    Конвертирует байты изображения в OpenCV формат (BGR)
    
    Args:
        image_bytes: изображение в виде байтов
        
    Returns:
        np.ndarray: изображение в формате OpenCV (BGR)
        
    Raises:
        Exception: при ошибке обработки изображения
    """
    try:
        buffer = io.BytesIO(image_bytes)
        pil_image = Image.open(buffer)
        rgb_array = np.array(pil_image)
        bgr_image = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        return bgr_image
    except Exception as e:
        raise Exception(f"Ошибка конвертации изображения: {e}")

def resize_images_to_match(image1: np.ndarray, image2: np.ndarray, 
                          target_size: Tuple[int, int] = (700, 500)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Изменяет размер двух изображений до одинакового размера
    
    Args:
        image1: первое изображение
        image2: второе изображение
        target_size: целевой размер (ширина, высота)
        
    Returns:
        Tuple: два изображения одинакового размера
    """
    resized_image1 = cv2.resize(image1, target_size)
    resized_image2 = cv2.resize(image2, target_size)
    return resized_image1, resized_image2

def blend_images(image1: np.ndarray, image2: np.ndarray, 
                alpha: float = 0.5, beta: float = 0.5, gamma: float = 0) -> np.ndarray:
    """
    Смешивает два изображения с заданными весами
    
    Args:
        image1: первое изображение
        image2: второе изображение
        alpha: вес первого изображения (0.0-1.0)
        beta: вес второго изображения (0.0-1.0)
        gamma: значение яркости
        
    Returns:
        np.ndarray: смешанное изображение
    """
    if image1.shape != image2.shape:
        raise ValueError("Изображения должны иметь одинаковый размер")
    
    if not (0 <= alpha <= 1 and 0 <= beta <= 1):
        raise ValueError("Веса alpha и beta должны быть в диапазоне [0, 1]")
    
    return cv2.addWeighted(image1, alpha, image2, beta, gamma)

def create_comparison_plot(image1: np.ndarray, image2: np.ndarray, 
                          blended_image: np.ndarray, alpha: float, beta: float,
                          output_path: Optional[str] = None, 
                          figsize: Tuple[int, int] = (30, 10)) -> None:
    """
    Создает сравнительный график трех изображений
    
    Args:
        image1: первое изображение
        image2: второе изображение
        blended_image: смешанное изображение
        alpha: вес первого изображения
        beta: вес второго изображения
        output_path: путь для сохранения результата
        figsize: размер фигуры
    """
    plt.figure(figsize=figsize)
    
    # Первое изображение
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    plt.title(f'Image 1, weight: {alpha}')
    plt.axis('off')
    
    # Второе изображение
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    plt.title(f'Image 2, weight: {beta}')
    plt.axis('off')
    
    # Смешанное изображение
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(blended_image, cv2.COLOR_BGR2RGB))
    plt.title(f'Blended Image, {alpha}/{beta}')
    plt.axis('off')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Результат сохранен в: {output_path}")
    
    plt.show()

def create_double_exposure(url1: str, url2: str, 
                          alpha: float = 0.7, beta: float = 0.3, 
                          target_size: Tuple[int, int] = (700, 500),
                          output_path: str = 'double_exposure_result.png') -> None:
    """
    Основная функция для создания двойной экспозиции
    
    Args:
        url1: URL первого изображения
        url2: URL второго изображения
        alpha: вес первого изображения (0.0-1.0)
        beta: вес второго изображения (0.0-1.0)
        target_size: размер выходных изображений
        output_path: путь для сохранения результата
    """
    try:
        # Загрузка изображений
        print("Загрузка изображений...")
        image1_bytes = download_image(url1)
        image2_bytes = download_image(url2)
        
        # Конвертация в OpenCV формат
        print("Обработка изображений...")
        cv2_image1 = bytes_to_cv2_image(image1_bytes)
        cv2_image2 = bytes_to_cv2_image(image2_bytes)
        
        # Изменение размера
        resized_image1, resized_image2 = resize_images_to_match(
            cv2_image1, cv2_image2, target_size
        )
        
        # Смешивание изображений
        print("Смешивание изображений...")
        blended_image = blend_images(resized_image1, resized_image2, alpha, beta)
        
        # Создание и сохранение результата
        print("Создание результата...")
        create_comparison_plot(
            resized_image1, resized_image2, blended_image, 
            alpha, beta, output_path
        )
        
        print("Готово! Двойная экспозиция создана успешно.")
        
    except Exception as e:
        print(f"Ошибка при создании двойной экспозиции: {e}")
        sys.exit(1)


if __name__ == "__main__":
    print("Введите URL первого изображения: ")
    url1 = input()  # "https://wallpapers.com/images/hd/iori-kitahara-standing-on-the-ledge-of-the-ocean-o1vsmby83sz5y57z.jpg"
    
    print("Введите URL второго изображения: ")
    url2 = input()  # "https://encrypted-tbn2.gstatic.com/images?q=tbn:ANd9GcQLVI5oqw867VkKFwDFtPaTGx2iRyLI2xth6m96yXdszR9vcfGi"
    
    print("Введите вес первого изображения (alpha, 0.0-1.0): ")
    alpha = float(input())
    
    print("Введите вес второго изображения (beta, 0.0-1.0): ")
    beta = float(input())
    
    print("Введите ширину изображения: ")
    width = int(input())
    
    print("Введите высоту изображения: ")
    height = int(input())
    
    print("Введите имя выходного файла: ")
    output_path = input()
    
    # Создание двойной экспозиции
    create_double_exposure(
        url1=url1,              # URL первого изображения
        url2=url2,              # URL второго изображения
        alpha=alpha,            # Вес первого изображения
        beta=beta,              # Вес второго изображения
        target_size=(width, height),  # Размер изображения
        output_path=output_path if output_path else 'my_double_exposure_result.png'
    )