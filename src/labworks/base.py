from abc import ABC, abstractmethod


class LabWork(ABC):
    """Базовый класс для всех лабораторных работ"""

    @abstractmethod
    def run(self):
        """Метод для выполнения лабораторной работы"""
        pass
