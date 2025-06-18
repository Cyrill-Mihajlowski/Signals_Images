from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from .base import LabWork


class Lab3(LabWork):
    """Лабораторная работа №3: Градационные алгоритмы обработки изображений"""

    def __init__(self, filepath: str):
        self.filepath = filepath

    def run(self):
        try:
            # Чтение изображения
            image, mode = self.read_image(self.filepath, "color")

            # Выбор преобразования
            self.display_menu()
            choice = input("Введите номер преобразования: ")

            # Отображение исходного изображения
            self.display_image(image, title="Исходное изображение", color_mode=mode)

            if choice == "1":
                gray_image = self.convert_to_grayscale(image)
                self.display_image(
                    gray_image, title="Полутоновое изображение", color_mode="grayscale"
                )
            elif choice == "2":
                brightness_image = self.adjust_brightness(image, 150)
                self.display_image(
                    brightness_image,
                    title="Изменение яркости изображения на 150 единиц",
                    color_mode=mode,
                )
            elif choice == "3":
                negative_image = self.convert_to_negative(image)
                self.display_image(
                    negative_image, title="Негатив изображения", color_mode=mode
                )
            elif choice == "4":
                gray_image = self.convert_to_grayscale(image)
                binary_image = self.convert_to_binary(gray_image, threshold=127)
                self.display_image(
                    binary_image, title="Бинарное изображение", color_mode="grayscale"
                )
            elif choice == "5":
                log_image = self.apply_log_transform(image)
                self.display_image(
                    log_image, title="Логарифмическое преобразование", color_mode=mode
                )
            elif choice == "6":
                gamma = float(input("Введите значение гаммы (например, 2.0): "))
                power_image = self.apply_power_transform(image, gamma=gamma)
                self.display_image(
                    power_image,
                    title=f"Степенное преобразование (гамма={gamma})",
                    color_mode=mode,
                )
            else:
                print("Неверный выбор!")

        except Exception as e:
            print(f"Ошибка: {e}")

    def convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Преобразование цветного изображения в полутоновое.

        :param image: Цветное изображение (BGR).
        :return: Полутоновое изображение.
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def adjust_brightness(self, image: np.ndarray, beta: int) -> np.ndarray:
        """
        Изменение яркости изображения.

        :param image: Исходное изображение как массив NumPy.
        :param beta: Добавочное значение яркости
        (целое число, положительное или отрицательное).
        :return: Изображение с измененной яркостью.
        """
        adjusted = cv2.convertScaleAbs(image, alpha=1, beta=beta)
        return adjusted

    def convert_to_negative(self, image: np.ndarray) -> np.ndarray:
        """
        Преобразование изображения в негатив.

        :param image: Исходное изображение.
        :return: Негатив изображения.
        """
        return 255 - image

    def convert_to_binary(self, image: np.ndarray, threshold: int = 127) -> np.ndarray:
        """
        Преобразование полутонового изображения в бинарное.

        :param image: Полутоновое изображение (grayscale).
        :param threshold: Пороговое значение (0-255).
        :return: Бинарное изображение.
        """
        _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        return binary_image

    def apply_power_transform(
        self, image: np.ndarray, gamma: float = 1.0
    ) -> NDArray[np.uint8]:
        """
        Степенное преобразование изображения.

        :param image: Исходное изображение.
        :param gamma: Показатель степени (r).
        :return: Изображение после степенного преобразования.
        """
        normalized = image / 255.0  # Нормализуем в диапазон [0, 1]
        power_image = np.power(normalized, gamma) * 255
        return np.uint8(power_image)  # type: ignore

    def apply_log_transform(self, image: np.ndarray) -> np.ndarray:
        """
        Логарифмическое преобразование изображения.

        :param image: Исходное изображение.
        :return: Изображение после логарифмического преобразования.
        """
        image = image.astype(
            np.float32
        )  # Преобразуем к float для предотвращения переполнения
        c = (
            255 / np.log(1 + np.max(image)) if np.max(image) > 0 else 1
        )  # Коэффициент нормализации
        log_image = c * np.log(1 + image)
        log_image = np.nan_to_num(
            log_image, nan=0.0, posinf=255, neginf=0
        )  # Заменяем NaN и inf
        return np.uint8(np.clip(log_image, 0, 255))  # type: ignore # Ограничиваем значения в диапазоне [0, 255]

    def read_image(self, filepath: str, color_mode: str = "color") -> Tuple:
        """
        Чтение изображения с заданным цветовым режимом.

        :param filepath: Путь к файлу изображения.
        :param color_mode: Режим чтения ('color', 'grayscale').
        :return: Изображение как массив NumPy и цветовой режим.
        """
        if color_mode == "color":
            image = cv2.imread(filepath, cv2.IMREAD_COLOR)
        elif color_mode == "grayscale":
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        else:
            raise ValueError(
                "Неверный режим чтения. Используйте 'color' или 'grayscale'."
            )

        if image is None:
            raise FileNotFoundError(f"Не удалось загрузить изображение: {filepath}")

        return image, color_mode

    def display_image(
        self, image, title: str = "Изображение", color_mode: str = "color"
    ) -> None:
        """
        Отображение изображения.

        :param image: Изображение как массив NumPy.
        :param title: Заголовок окна.
        :param color_mode: Цветовой режим ('color', 'grayscale').
        """
        plt.figure(figsize=(8, 6))
        if color_mode == "color":
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        elif color_mode == "grayscale":
            plt.imshow(image, cmap="gray")
        else:
            raise ValueError("Неверный цветовой режим.")
        plt.title(title)
        plt.axis("off")
        plt.show()

    def display_menu(self):
        print(
            "Выберите преобразование:"
            "\n 1. Преобразование в полутоновое"
            "\n 2. Изменение яркости изображения на 150 единиц"
            "\n 3. Преобразование изображения в негатив"
            "\n 4. Преобразование в бинарное"
            "\n 5. Логарифмическое преобразование"
            "\n 6. Степенное преобразование"
        )
