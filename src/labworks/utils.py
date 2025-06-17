import matplotlib.pyplot as plt
import numpy as np


def plot_signal(
    time: np.ndarray,
    signal: np.ndarray,
    title: str,
    xlabel: str = "Время (с)",
    ylabel: str = "Амплитуда",
    max_points: int = 1000,
    grid: bool = True,
) -> None:
    """Визуализация сигнала во временной области

    Args:
        time (_np.ndarray_): Временной массив (в секундах)
        signal (_np.ndarray_): Сигнал для отображения
        title (_str_): Заголовок графика
        xlabel (_str_): Надпись на Оси X
        ylabel (_str_): Надпись на Оси Y
        max_points (int, optional): Максимальное количество точек для отображения
        Defaults to 1000.
    """

    plt.figure(figsize=(10, 4))

    plt.plot(time[:max_points], signal[:max_points])

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if grid:
        plt.grid()
    plt.show()
    plt.close("all")


def plot_frequency_response(signal_data, sample_rate):
    """Визуализация амплитудно-частотной характеристики (АЧХ)"""

    fft = np.fft.fft(signal_data)
    freqs = np.fft.fftfreq(len(fft), 1 / sample_rate)
    plt.figure(figsize=(10, 4))
    plt.plot(freqs[: len(freqs) // 2], np.abs(fft)[: len(freqs) // 2])
    plt.xlabel("Частота (Гц)")
    plt.ylabel("Амплитуда")
    plt.title("Амплитудно-частотная характеристика (АЧХ)")
    plt.show()
    plt.close("all")  # Закрытие фигуры


def plot_acf(
    signal_data: np.ndarray,
    max_lags: int = 500,  # type: ignore
    title: str = "Автокорреляционная функция (АКФ)",
) -> None:
    """
    Визуализация автокорреляционной функции (АКФ).

    Args:
        signal_data (np.ndarray): Сигнал для анализа.
        max_lags (int, optional): Максимальное количество лагов для отображения.
        Если None, используется полный диапазон.
        title (str, optional): Заголовок графика.
    """
    if len(signal_data) == 0:
        raise ValueError("Сигнал пуст. Невозможно построить АКФ.")

    acf = np.correlate(signal_data, signal_data, mode="full")  # Расчет АКФ
    lags = np.arange(-len(signal_data) + 1, len(signal_data))

    if max_lags:
        center = len(signal_data) - 1
        acf = acf[center - max_lags : center + max_lags + 1]
        lags = lags[center - max_lags : center + max_lags + 1]

    plt.figure(figsize=(10, 4))
    plt.plot(lags, acf)
    plt.title(title)
    plt.xlabel("Лаг")
    plt.ylabel("Амплитуда")
    plt.grid()
    plt.show()
    plt.close("all")
