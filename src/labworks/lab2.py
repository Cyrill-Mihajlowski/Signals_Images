import numpy as np
import scipy.signal as signal

from .base import LabWork
from .utils import plot_signal


class Lab2(LabWork):
    """Лабораторная работа №2"""

    def run(self):
        print(
            "Выберите тип сигнала: \n1. Гармонический сигнал "
            "\n2. Белый шум "
            "\n3. Полигармонический сигнал "
            "\n4. Узкополосный шум "
            "\n5. Амплитудно-модулированный сигнал"
        )

        signal_type = input("Введите номер сигнала (1-5): ")

        sample_rate = 44100  # Частота дискретизации
        duration = 5  # Длительность сигнала в секундах

        if signal_type == "1":
            signal = self.generate_signal("sine", duration, sample_rate)
            title = "Гармонический сигнал"
        elif signal_type == "2":
            signal = self.generate_signal("white_noise", duration, sample_rate)
            title = "Белый шум"
        elif signal_type == "3":
            frequencies = [100, 300, 600]
            amplitudes = [1.0, 0.5, 0.25]
            signal = self.generate_polyharmonic_signal(
                frequencies, amplitudes, duration, sample_rate
            )
            title = "Полигармонический сигнал"
        elif signal_type == "4":
            signal = self.generate_bandlimited_noise(500, 200, duration, sample_rate)
            title = "Узкополосный шум"
        elif signal_type == "5":
            signal = self.generate_am_signal(1000, 10, 0.5, duration, sample_rate)
            title = "Амплитудно-модулированный сигнал"
        else:
            print("Неверный выбор.")
            return

        print(f"Сгенерирован сигнал с {len(signal)} отсчетами.")

        # Визуализация сигнала
        time = np.linspace(0, duration, len(signal), endpoint=False)
        plot_signal(time, signal, title)

    def generate_signal(self, signal_type, duration, sample_rate):
        """Генерация синтетического сигнала.

        :param signal_type: Тип сигнала ('sine', 'white_noise', и т.д.)
        :param duration: Длительность сигнала (в секундах)
        :param sample_rate: Частота дискретизации (Гц)
        :return: Массив значений сигнала
        """

        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

        if signal_type == "sine":
            frequency = 440  # Частота гармонического сигнала, Гц
            return np.sin(2 * np.pi * frequency * t)
        elif signal_type == "white_noise":
            return np.random.normal(0, 1, len(t))
        else:
            raise ValueError(f"Неизвестный тип сигнала: {signal_type}")

    def compute_signal_parameters(self, signal):
        """Вычисление статистических параметров сигнала.

        :param signal: Сигнал (массив значений)
        :return: Словарь с параметрами сигнала
        """

        power = np.mean(signal**2)  # Средняя мощность
        mean = np.mean(signal)  # Математическое ожидание
        variance = np.var(signal)  # Дисперсия
        std_dev = np.sqrt(variance)  # Среднеквадратическое отклонение

        return {
            "Мощность": power,
            "Среднее": mean,
            "Дисперсия": variance,
            "СКО": std_dev,
        }

    def generate_polyharmonic_signal(
        self, frequencies, amplitudes, duration, sample_rate, noise_amplitude=0.0
    ):
        """
        Генерация полигармонического сигнала с добавлением белого шума.

        :param frequencies: Список частот гармонических сигналов
        :param amplitudes: Список амплитуд гармонических сигналов
        :param duration: Длительность сигнала (в секундах)
        :param sample_rate: Частота дискретизации (Гц)
        :param noise_amplitude: Амплитуда белого шума (по умолчанию 0.0)
        :return: Сигнал в виде массива
        """
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        signal = np.zeros_like(t)

        # Генерация гармоник
        for freq, amp in zip(frequencies, amplitudes, strict=False):
            signal += amp * np.sin(2 * np.pi * freq * t)

        # Добавление белого шума
        if noise_amplitude > 0.0:
            noise = np.random.normal(0, noise_amplitude, len(t))
            signal += noise

        return signal

    def generate_bandlimited_noise(
        self, center_freq: float, bandwidth: float, duration: float, sample_rate: int
    ) -> np.ndarray:
        """
        Генерация узкополосного шума.

        :param center_freq: Центральная частота шума (Гц).
        :param bandwidth: Полоса пропускания шума (Гц).
        :param duration: Длительность сигнала (с).
        :param sample_rate: Частота дискретизации (Гц).
        :return: Узкополосный шум.
        """
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

        # Генерация белого шума
        white_noise = np.random.normal(0, 1, len(t))

        # Создание полосового фильтра
        nyquist = 0.5 * sample_rate
        low_cutoff = (center_freq - bandwidth / 2) / nyquist
        high_cutoff = (center_freq + bandwidth / 2) / nyquist

        b, a = signal.butter(4, [low_cutoff, high_cutoff], btype="band")  # type: ignore
        bandlimited_noise = signal.filtfilt(b, a, white_noise)

        return bandlimited_noise

    def generate_am_signal(
        self,
        carrier_freq: float,
        modulating_freq: float,
        mod_index: float,
        duration: float,
        sample_rate: int,
    ) -> np.ndarray:
        """
        Генерация амплитудно-модулированного сигнала.

        :param carrier_freq: Частота несущей (Гц).
        :param modulating_freq: Частота модуляции (Гц).
        :param mod_index: Индекс модуляции (0-1).
        :param duration: Длительность сигнала (с).
        :param sample_rate: Частота дискретизации (Гц).
        :return: Амплитудно-модулированный сигнал.
        """
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

        carrier = np.sin(2 * np.pi * carrier_freq * t)
        modulating = np.sin(2 * np.pi * modulating_freq * t)

        am_signal = (1 + mod_index * modulating) * carrier
        return am_signal
