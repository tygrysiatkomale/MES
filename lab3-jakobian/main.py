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

# Funkcja główna - przetwarza wszystkie punkty całkowania
def oblicz_jakobian(plik):
    ksi_wsp, eta_wsp = czytaj_wspolrzedne(plik)

    # 4-punktowa kwadratura Gaussa w 1D
    punkty_1D = [-np.sqrt(3 / 7 + (2 / 7 * np.sqrt(6 / 5))),
                 -np.sqrt(3 / 7 - (2 / 7 * np.sqrt(6 / 5))),
                 np.sqrt(3 / 7 - (2 / 7 * np.sqrt(6 / 5))),
                 np.sqrt(3 / 7 + (2 / 7 * np.sqrt(6 / 5)))]

    wagi_1D = [(18 - np.sqrt(30)) / 36,
               (18 + np.sqrt(30)) / 36,
               (18 + np.sqrt(30)) / 36,
               (18 - np.sqrt(30)) / 36]

    # Tworzymy siatkę punktów Gaussa dla 2D (iloczyn kartezjański)
    punkty_calek = [(ksi, eta) for ksi in punkty_1D for eta in punkty_1D]
    wagi_calek = [w1 * w2 for w1 in wagi_1D for w2 in wagi_1D]

    # Iteracja przez punkty całkowania
    for i, (ksi_pkt, eta_pkt) in enumerate(punkty_calek):
        dN_dksi, dN_deta = pochodne_funkcji_ksztaltu(ksi_pkt, eta_pkt)

        for elem in range(len(ksi_wsp)):  # Iteracja przez elementy
            wsp_x = ksi_wsp[elem]
            wsp_y = eta_wsp[elem]

            # Obliczenie Jakobianu w punkcie całkowania
            J = jakobian(wsp_x, wsp_y, dN_dksi, dN_deta)
            detJ = det_jakobianu(J)

            if detJ != 0:
                J_inv = odwrotnosc_jakobianu(J)
            else:
                J_inv = None

            # Wyświetlenie wyników
            print(f"Element {elem+1}, Punkt całkowania {i+1}: ({ksi_pkt}, {eta_pkt})")
            print(f"Jakobian [J]:\n{J}")
            print(f"Wyznacznik det[J]: {detJ}")
            if J_inv is not None:
                print(f"Odwrotność [J^-1]:\n{J_inv}")
            else:
                print("Jakobian jest osobliwy, brak odwrotności.")
            print("\n")

# Przykład użycia funkcji - załóżmy, że mamy plik 'wspolrzedne.txt'
oblicz_jakobian('wspolrzedne.txt')
