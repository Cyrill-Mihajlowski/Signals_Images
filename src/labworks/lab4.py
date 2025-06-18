from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .base import LabWork


class Lab4(LabWork):
    """Лабораторная работа №4: Алгоритмы фильтрации изображений"""

    def __init__(self, filepath: str):
        self.filepath = filepath

    def run(self):
        try:
            # Чтение изображения
            image, mode = self.read_image(self.filepath, "color")

            # Отображение исходного изображения
            self.display_image(image, title="Исходное изображение", color_mode=mode)

            # Выбор фильтра
            print("Выберите фильтр:")
            print("1. Низкочастотный фильтр (сглаживание)")
            print("2. Высокочастотный фильтр (усиление границ)")
            print("3. Медианный фильтр")
            choice = input("Введите номер фильтра (1-3): ")

            if choice == "1":
                kernel_size = int(input("Введите размер маски (например, 3): "))
                filtered_image = self.apply_low_pass_filter(image, kernel_size)
                self.display_image(filtered_image, title="НЧ фильтр", color_mode=mode)
            elif choice == "2":
                kernel_size = int(input("Введите размер маски (например, 3): "))
                filtered_image = self.apply_high_pass_filter(image, kernel_size)
                self.display_image(filtered_image, title="ВЧ фильтр", color_mode=mode)
            elif choice == "3":
                kernel_size = int(input("Введите размер окна (например, 3): "))
                filtered_image = self.apply_median_filter(image, kernel_size)
                self.display_image(
                    filtered_image, title="Медианный фильтр", color_mode=mode
                )
            else:
                print("Неверный выбор!")
        except Exception as e:
            print(f"Ошибка: {e}")

    def apply_low_pass_filter(self, image: np.ndarray, kernel_size: int) -> np.ndarray:
        """
        Низкочастотный фильтр (сглаживание).

        :param image: Исходное изображение.
        :param kernel_size: Размер ядра (нечетное число).
        :return: Отфильтрованное изображение.
        """
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
        return cv2.filter2D(image, -1, kernel)

    def apply_high_pass_filter(self, image: np.ndarray, kernel_size: int) -> np.ndarray:
        """
        Высокочастотный фильтр (усиление границ).

        :param image: Исходное изображение.
        :param kernel_size: Размер ядра (нечетное число).
        :return: Отфильтрованное изображение.
        """
        kernel = -np.ones((kernel_size, kernel_size), np.float32)
        kernel[kernel_size // 2, kernel_size // 2] = kernel_size**2 - 1
        return cv2.filter2D(image, -1, kernel)

    def apply_median_filter(self, image: np.ndarray, kernel_size: int) -> np.ndarray:
        """
        Медианный фильтр.

        :param image: Исходное изображение.
        :param kernel_size: Размер окна (нечетное число).
        :return: Отфильтрованное изображение.
        """
        return cv2.medianBlur(image, kernel_size)

    def read_image(self, filepath: str, color_mode: str = "color") -> Tuple:
        """Чтение изображения."""
        if color_mode == "color":
            image = cv2.imread(filepath, cv2.IMREAD_COLOR)
        elif color_mode == "grayscale":
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        else:
            raise ValueError("Неверный режим чтения!")
        if image is None:
            raise FileNotFoundError(f"Файл не найден: {filepath}")
        return image, color_mode

    def display_image(
        self, image: np.ndarray, title: str = "Изображение", color_mode: str = "color"
    ):
        """Отображение изображения."""
        plt.figure(figsize=(8, 6))
        if color_mode == "color":
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        elif color_mode == "grayscale":
            plt.imshow(image, cmap="gray")
        else:
            raise ValueError("Неверный цветовой режим!")
        plt.title(title)
        plt.axis("off")
        plt.show()
