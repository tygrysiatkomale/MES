import numpy as np


def kwadratura_gaussa_2D(func, schemat):
    if schemat == 2:
        # 2-punktowa kwadratura Gaussa
        punkty = [-np.sqrt(1 / 3), np.sqrt(1 / 3)]
        wagi = [1, 1]
    elif schemat == 3:
        # 3-punktowa kwadratura Gaussa
        punkty = [-np.sqrt(3 / 5), 0, np.sqrt(3 / 5)]
        wagi = [5 / 9, 8 / 9, 5 / 9]
    elif schemat == 4:
        # 4-punktowa kwadratura Gaussa
        punkty = [-np.sqrt(3 / 7 + (2 / 7 * np.sqrt(6 / 5))),
                  -np.sqrt(3 / 7 - (2 / 7 * np.sqrt(6 / 5))),
                  np.sqrt(3 / 7 - (2 / 7 * np.sqrt(6 / 5))),
                  np.sqrt(3 / 7 + (2 / 7 * np.sqrt(6 / 5)))]
        wagi = [(18 - np.sqrt(30)) / 36,
                (18 + np.sqrt(30)) / 36,
                (18 + np.sqrt(30)) / 36,
                (18 - np.sqrt(30)) / 36]
    elif schemat == 5:
        # 5-punktowa kwadratura Gaussa
        punkty = [-np.sqrt((5 + 2 * np.sqrt(10 / 7)) / 9),
                  -np.sqrt((5 - 2 * np.sqrt(10 / 7)) / 9),
                  0,
                  np.sqrt((5 - 2 * np.sqrt(10 / 7)) / 9),
                  np.sqrt((5 + 2 * np.sqrt(10 / 7)) / 9)]
        wagi = [(322 - 13 * np.sqrt(70)) / 900,
                (322 + 13 * np.sqrt(70)) / 900,
                128 / 225,
                (322 + 13 * np.sqrt(70)) / 900,
                (322 - 13 * np.sqrt(70)) / 900]
    else:
        raise ValueError("Niepoprawny schemat całkowania. Wybierz 2 lub 3.")

    wynik = 0
    for i, xi in enumerate(punkty):
        for j, yj in enumerate(punkty):
            wynik += wagi[i] * wagi[j] * func(xi, yj)

    return wynik


def f(x, y):
    return -2 * x**2 * y + 2 * x * y + 4


result_2 = kwadratura_gaussa_2D(f, 2)
print(result_2)

result_3 = kwadratura_gaussa_2D(f, 3)
print(result_3)

result_4 = kwadratura_gaussa_2D(f, 4)
print(result_4)

result_5 = kwadratura_gaussa_2D(f, 5)
print(result_5)

