import numpy as np
import scipy.signal as signal
import soundfile as sf

from labworks.utils import plot_acf, plot_frequency_response, plot_signal

from .base import LabWork


class Lab1(LabWork):
    """Лабораторная работа №1"""

    def __init__(self, filepath: str):
        self.filepath = filepath

    def run(self):
        # Чтение WAV-файла
        sample_rate, data = self.read_wav(self.filepath)

        # Ограничиваем длину сигнала для ускорения
        if len(data) > 500000:
            print("Сигнал слишком длинный, обрезаем до 500000 отсчетов.")
            data = data[:500000]

        time = np.linspace(0, len(data) / sample_rate, num=len(data))

        # Визуализация исходного сигнала
        plot_signal(time=time, signal=data, title="Исходный сигнал", max_points=1000000)

        # Генерация синтетического (гармонический) сигнала
        generated_signal = self.generate_signal(
            "sine_wave", duration=5, sample_rate=sample_rate
        )

        plot_signal(
            time=np.linspace(0, 5, len(generated_signal)),
            signal=generated_signal,
            title="Гармонический сигнал",
        )

        # Вычисление параметров сигнала
        power, mean, variance = self.compute_parameters(data)
        print(f"Мощность: {power}, Среднее: {mean}, Дисперсия: {variance}")

        # Фильтрация сигнала
        filtered_signal = self.apply_filter(
            data, filter_type="low", cutoff=500, sample_rate=sample_rate
        )

        plot_signal(
            time=time,
            signal=filtered_signal,
            title="Фильтрованный сигнал",
            max_points=1000000000,
        )

        # Визуализация амплитудно-частотной характеристики (АЧХ)
        plot_frequency_response(data, sample_rate)

        # Визуализация автокорреляционной функции (АКФ)
        plot_acf(data)

    def read_wav(self, filepath: str) -> tuple[int, np.ndarray]:
        """Чтение WAV-файла с использованием soundfile.

        Args:
            filepath (str): Путь к файлу.

        Returns:
            tuple[int, np.ndarray]: Частота дискретизации и данные сигнала.
        """

        data, sample_rate = sf.read(filepath)
        if len(data.shape) > 1:
            data = data[:, 0]  # Выбираем первый канал
        return sample_rate, data

    def generate_signal(self, signal_type, duration, sample_rate):
        """Генерация синтетического сигнала"""

        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        if signal_type == "white_noise":
            return np.random.normal(0, 1, len(t))
        elif signal_type == "sine_wave":
            frequency = 440  # Частота гармонического сигнала
            return np.sin(2 * np.pi * frequency * t)
        elif signal_type == "pulse":
            # Пример импульсного сигнала
            return signal.square(2 * np.pi * 1 * t)
        else:
            raise ValueError("Неизвестный тип сигнала")

    def compute_parameters(self, signal):
        """
        Вычисление основных параметров сигнала
        """

        power = np.mean(signal**2)  # Средняя мощность
        mean = np.mean(signal)  # Математическое ожидание
        variance = np.var(signal)  # Дисперсия
        return power, mean, variance

    def apply_filter(
        self, signal_data: np.ndarray, filter_type: str, cutoff: float, sample_rate: int
    ) -> np.ndarray:
        """Применение цифрового фильтра к сигналу.

        Args:
            signal_data (np.ndarray): Сигнал для фильтрации.
            filter_type (str): Тип фильтра ('low', 'high', 'bandpass', 'bandstop').
            cutoff (float): Частота среза (Гц).
            sample_rate (int): Частота дискретизации (Гц).

        Returns:
            np.ndarray: Отфильтрованный сигнал.
        """

        nyquist = 0.5 * sample_rate
        norm_cutoff = cutoff / nyquist

        if not (0 < norm_cutoff < 1):
            raise ValueError(
                "Частота среза должна быть в диапазоне"
                f"(0, {nyquist}). Получено: {cutoff}"
            )

        if filter_type not in ["low", "high", "bandpass", "bandstop"]:
            raise ValueError(
                f"Тип фильтра должен быть одним из: \
                ['low', 'high', 'bandpass', 'bandstop']. "
                f"Получено: {filter_type}"
            )

        try:
            b, a = signal.butter(4, norm_cutoff, btype=filter_type, analog=False)  # type: ignore
            return signal.filtfilt(b, a, signal_data)
        except Exception as e:
            raise RuntimeError(f"Ошибка при применении фильтра: {e}")  # noqa: B904
