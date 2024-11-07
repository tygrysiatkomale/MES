import numpy as np


# Funkcja do wczytywania współrzędnych węzłów z pliku
def czytaj_wspolrzedne(plik):
    ksi = []
    eta = []
    with open(plik, 'r') as f:
        for line in f:
            values = list(map(float, line.split()))
            ksi.append(values[0:4])  # Zakładamy, że mamy 4 węzły dla każdego elementu
            eta.append(values[4:8])  # Kolejne 4 wartości to eta dla tych samych węzłów
    return np.array(ksi), np.array(eta)


# Funkcje kształtu dla 4-węzłowego elementu kwadratowego
def funkcje_ksztaltu(ksi, eta):
    N1 = 0.25 * (1 - ksi) * (1 - eta)
    N2 = 0.25 * (1 + ksi) * (1 - eta)
    N3 = 0.25 * (1 + ksi) * (1 + eta)
    N4 = 0.25 * (1 - ksi) * (1 + eta)
    return np.array([N1, N2, N3, N4])


# Pochodne funkcji kształtu względem ksi i eta
def pochodne_funkcji_ksztaltu(ksi, eta):
    dN_dksi = [-0.25 * (1 - eta), 0.25 * (1 - eta), 0.25 * (1 + eta), -0.25 * (1 + eta)]
    dN_deta = [-0.25 * (1 - ksi), -0.25 * (1 + ksi), 0.25 * (1 + ksi), 0.25 * (1 - ksi)]
    return np.array(dN_dksi), np.array(dN_deta)


# Funkcja do obliczania Jakobianu w punkcie całkowania
def jakobian(wsp_x, wsp_y, dN_dksi, dN_deta):
    J = np.zeros((2, 2))
    J[0, 0] = np.dot(dN_dksi, wsp_x)  # ∂x/∂ξ
    J[0, 1] = np.dot(dN_deta, wsp_x)  # ∂x/∂η
    J[1, 0] = np.dot(dN_dksi, wsp_y)  # ∂y/∂ξ
    J[1, 1] = np.dot(dN_deta, wsp_y)  # ∂y/∂η
    return J


# Funkcja do obliczania wyznacznika Jakobianu
def det_jakobianu(J):
    return np.linalg.det(J)


# Funkcja do obliczania odwrotności Jakobianu
def odwrotnosc_jakobianu(J):
    return np.linalg.inv(J)


# Funkcja do obliczania macierzy H lokalnej
def oblicz_macierz_H_lokalna(wsp_x, wsp_y, punkty_calek, wagi_calek, k):
    H = np.zeros((4, 4))  # Inicjalizacja lokalnej macierzy H
    for i, (ksi_pkt, eta_pkt) in enumerate(punkty_calek):
        waga = wagi_calek[i]
        dN_dksi, dN_deta = pochodne_funkcji_ksztaltu(ksi_pkt, eta_pkt)
        J = jakobian(wsp_x, wsp_y, dN_dksi, dN_deta)
        detJ = det_jakobianu(J)
        if detJ == 0:
            continue  # Pomijamy punkt, jeśli wyznacznik Jacobiego jest zerowy
        J_inv = odwrotnosc_jakobianu(J)

        # Obliczanie pochodnych względem x i y
        dN_dx = J_inv[0, 0] * dN_dksi + J_inv[0, 1] * dN_deta
        dN_dy = J_inv[1, 0] * dN_dksi + J_inv[1, 1] * dN_deta

        # Składanie macierzy H
        H_local = k * (np.outer(dN_dx, dN_dx) + np.outer(dN_dy, dN_dy)) * detJ * waga
        H += H_local
    return H


# Funkcja główna - przetwarza wszystkie elementy i oblicza macierz H
def oblicz_H_dla_elementow(plik, k):
    ksi_wsp, eta_wsp = czytaj_wspolrzedne(plik)

    # Definiowanie punktów Gaussa i wag
    punkty_1D = [-np.sqrt(3 / 7 + (2 / 7 * np.sqrt(6 / 5))),
                 -np.sqrt(3 / 7 - (2 / 7 * np.sqrt(6 / 5))),
                 np.sqrt(3 / 7 - (2 / 7 * np.sqrt(6 / 5))),
                 np.sqrt(3 / 7 + (2 / 7 * np.sqrt(6 / 5)))]
    wagi_1D = [(18 - np.sqrt(30)) / 36,
               (18 + np.sqrt(30)) / 36,
               (18 + np.sqrt(30)) / 36,
               (18 - np.sqrt(30)) / 36]

    # Tworzenie siatki punktów Gaussa dla 2D (iloczyn kartezjański)
    punkty_calek = [(ksi, eta) for ksi in punkty_1D for eta in punkty_1D]
    wagi_calek = [w1 * w2 for w1 in wagi_1D for w2 in wagi_1D]

    # Iteracja przez elementy
    for elem in range(len(ksi_wsp)):
        wsp_x = ksi_wsp[elem]
        wsp_y = eta_wsp[elem]

        # Obliczenie lokalnej macierzy H
        H = oblicz_macierz_H_lokalna(wsp_x, wsp_y, punkty_calek, wagi_calek, k)

        # Wyświetlenie macierzy H dla elementu
        print(f"Element {elem + 1}:")
        print(f"Lokalna macierz H:\n{H}\n")


# Przykładowe wywołanie
if __name__ == "__main__":
    k = 30.0  # Przykładowy współczynnik przewodzenia ciepła
    oblicz_H_dla_elementow('wspolrzedne.txt', k)
