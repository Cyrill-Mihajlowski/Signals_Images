from labworks.lab1 import Lab1
from labworks.lab2 import Lab2


def main() -> None:
    labs = {
        "1": Lab1(filepath="./res/test_1.wav"),
        "2": Lab2(),
    }

    print(
        "Методы и технологии обработки сигналов и изображений:\n"
        "1: Лабораторная работа №1 - Цифровая фильтрация сигналов.\n"
        "2: Лабораторная работа №2 - Методы статистического и "
        "спектрального оценивания сигналов."
    )

    choice = input("Выберите номер лабораторной работы: ")

    lab = labs.get(choice)
    if lab is None:
        print("Неверный выбор!")
        return lab

    lab.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n Завершение работы.")
