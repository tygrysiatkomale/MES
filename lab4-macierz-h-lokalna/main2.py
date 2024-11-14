import numpy as np


# Funkcja do wczytywania współrzędnych węzłów z pliku
def czytaj_wspolrzedne(plik):
    ksi = []
    eta = []
    with open(plik, 'r') as f:
        for line in f:
            values = list(map(float, line.split()))
            ksi.append(values[0:4])
            eta.append(values[4:8])
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
    J[1, 0] = np.dot(dN_deta, wsp_x)  # ∂x/∂η
    J[0, 1] = np.dot(dN_dksi, wsp_y)  # ∂y/∂ξ
    J[1, 1] = np.dot(dN_deta, wsp_y)  # ∂y/∂η
    return J


# Funkcja do obliczania wyznacznika Jakobianu
def det_jakobianu(J):
    return np.linalg.det(J)


# Funkcja do obliczania odwrotności Jakobianu
def odwrotnosc_jakobianu(J):
    return np.linalg.inv(J)


# Funkcja do obliczania macierzy H lokalnej z wypisywaniem kroków
def oblicz_macierz_H_lokalna(wsp_x, wsp_y, punkty_calek, wagi_calek, k):
    H = np.zeros((4, 4))  # Inicjalizacja lokalnej macierzy H
    for i, (ksi_pkt, eta_pkt) in enumerate(punkty_calek):
        waga = wagi_calek[i]
        print(f"\nPunkt całkowania {i + 1}: ksi = {ksi_pkt}, eta = {eta_pkt}, waga = {waga}")

        # Pochodne funkcji kształtu względem ksi i eta
        dN_dksi, dN_deta = pochodne_funkcji_ksztaltu(ksi_pkt, eta_pkt)
        print(f"Pochodne funkcji kształtu względem ksi: {dN_dksi}")
        print(f"Pochodne funkcji kształtu względem eta: {dN_deta}")

        # Obliczanie Jakobianu
        J = jakobian(wsp_x, wsp_y, dN_dksi, dN_deta)
        print(f"Jakobian J:\n{J}")

        # Wyznacznik Jakobianu
        detJ = det_jakobianu(J)
        print(f"Wyznacznik Jakobianu: {detJ}")

        if detJ == 0:
            print("Pominięto punkt całkowania ze względu na zerowy wyznacznik Jacobiego.")
            continue  # Pomijamy punkt, jeśli wyznacznik Jacobiego jest zerowy

        # Odwrotność Jakobianu
        J_inv = odwrotnosc_jakobianu(J)
        print(f"Odwrotność Jakobianu J_inv:\n{J_inv}")

        # Obliczanie pochodnych względem x i y
        dN_dx = J_inv[0, 0] * dN_dksi + J_inv[0, 1] * dN_deta
        dN_dy = J_inv[1, 0] * dN_dksi + J_inv[1, 1] * dN_deta
        print(f"Pochodne funkcji kształtu względem x: {dN_dx}")

        # Obliczanie macierzy H lokalnej w tym punkcie całkowania
        H_local = k * (np.outer(dN_dx, dN_dx) + np.outer(dN_dy, dN_dy)) * detJ * waga
        print(f"Macierz H lokalna dla punktu całkowania {i + 1}:\n{H_local}")

        # Dodanie H_local do całkowitej macierzy H
        H += H_local

    print(f"\nLokalna macierz H dla elementu:\n{H}\n")
    return H


# Funkcja główna - przetwarza wszystkie elementy i oblicza macierz H
def oblicz_H_dla_elementow(plik, k):
    ksi_wsp, eta_wsp = czytaj_wspolrzedne(plik)

    # Definiowanie punktów Gaussa i wag
    punkty_1D = [-1/np.sqrt(3), 1/np.sqrt(3)]
    wagi_1D = [1, 1]
    # Tworzenie siatki punktów Gaussa dla 2D (iloczyn kartezjański)
    punkty_calek = [(ksi, eta) for eta in punkty_1D for ksi in punkty_1D]
    wagi_calek = [w1 * w2 for w1 in wagi_1D for w2 in wagi_1D]

    # Iteracja przez elementy
    for elem in range(len(ksi_wsp)):
        wsp_x = ksi_wsp[elem]
        wsp_y = eta_wsp[elem]

        # Obliczenie lokalnej macierzy H
        print(f"\nElement {elem + 1}:")
        punkty_calek[2], punkty_calek[3] = punkty_calek[3], punkty_calek[2]
        H = oblicz_macierz_H_lokalna(wsp_x, wsp_y, punkty_calek, wagi_calek, k)


# Przykładowe wywołanie
if __name__ == "__main__":
    k = 30.0  # Przykładowy współczynnik przewodzenia ciepła
    oblicz_H_dla_elementow('wspolrzedne2.txt', k)
