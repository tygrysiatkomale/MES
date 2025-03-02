import numpy as np


# Funkcja do wczytywania danych z pliku siatki
def czytajDaneZPliku(plik):
    with open(plik, 'r') as f:
        lines = f.readlines()

    data = {}
    idx = 0
    # Czytamy parametry symulacji
    while idx < len(lines):
        line = lines[idx].strip()
        idx += 1
        if line.startswith("SimulationTime"):
            data["SimulationTime"] = float(line.split()[1])
        elif line.startswith("SimulationStepTime"):
            data["SimulationStepTime"] = float(line.split()[1])
        elif line.startswith("Conductivity"):
            data["Conductivity"] = float(line.split()[1])
        elif line.startswith("Alfa"):
            data["Alfa"] = float(line.split()[1])
        elif line.startswith("Tot"):
            data["Tot"] = float(line.split()[1])
        elif line.startswith("InitialTemp"):
            data["InitialTemp"] = float(line.split()[1])
        elif line.startswith("Density"):
            data["Density"] = float(line.split()[1])
        elif line.startswith("SpecificHeat"):
            data["SpecificHeat"] = float(line.split()[1])
        elif line.startswith("Nodes number"):
            data["NodesNumber"] = int(line.split()[2])
        elif line.startswith("Elements number"):
            data["ElementsNumber"] = int(line.split()[2])
        elif line.startswith("*Node"):
            # czytamy węzły
            nodes = []
            for _ in range(data["NodesNumber"]):
                l = lines[idx].strip()
                idx += 1
                parts = l.split(',')
                n_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                nodes.append((n_id, x, y))
            data["Nodes"] = nodes
        elif line.startswith("*Element"):
            # czytamy elementy
            elements = []
            for _ in range(data["ElementsNumber"]):
                l = lines[idx].strip()
                idx += 1
                parts = l.split(',')
                e_id = int(parts[0])
                e_nodes = list(map(int, parts[1:]))
                elements.append((e_id, e_nodes))
            data["Elements"] = elements
        elif line.startswith("*BC"):
            # czytamy węzły na których jest BC
            bc_nodes = []
            bc_line = lines[idx].strip()
            idx += 1
            bc_parts = bc_line.split(',')
            for b in bc_parts:
                if b.strip().isdigit():
                    bc_nodes.append(int(b.strip()))
            data["BC"] = bc_nodes

    return data


# Funkcje kształtu dla elementu 4-węzłowego (2D)
def funkcje_ksztaltu(ksi, eta):
    N1 = 0.25 * (1 - ksi) * (1 - eta)
    N2 = 0.25 * (1 + ksi) * (1 - eta)
    N3 = 0.25 * (1 + ksi) * (1 + eta)
    N4 = 0.25 * (1 - ksi) * (1 + eta)
    return np.array([N1, N2, N3, N4])


# Pochodne funkcji kształtu względem ksi i eta
def pochodne_funkcji_ksztaltu(ksi, eta):
    dN_dksi = np.array([
        -0.25 * (1 - eta),
        0.25 * (1 - eta),
        0.25 * (1 + eta),
        -0.25 * (1 + eta)
    ])
    dN_deta = np.array([
        -0.25 * (1 - ksi),
        -0.25 * (1 + ksi),
        0.25 * (1 + ksi),
        0.25 * (1 - ksi)
    ])
    return dN_dksi, dN_deta


# Jakobian
def jakobian(wsp_x, wsp_y, dN_dksi, dN_deta):
    J = np.zeros((2, 2))
    J[0, 0] = np.dot(dN_dksi, wsp_x)  # ∂x/∂ξ
    J[0, 1] = np.dot(dN_dksi, wsp_y)  # ∂y/∂ξ
    J[1, 0] = np.dot(dN_deta, wsp_x)  # ∂x/∂η
    J[1, 1] = np.dot(dN_deta, wsp_y)  # ∂y/∂η
    return J


def det_jakobianu(J):
    return np.linalg.det(J)


def odwrotnosc_jakobianu(J):
    return np.linalg.inv(J)


# Funkcja do obliczania macierzy H lokalnej
def oblicz_macierz_H_lokalna(wsp_x, wsp_y, punkty_calek, wagi_calek, k):
    H = np.zeros((4, 4))
    for i, (ksi_pkt, eta_pkt) in enumerate(punkty_calek):
        waga = wagi_calek[i]
        dN_dksi, dN_deta = pochodne_funkcji_ksztaltu(ksi_pkt, eta_pkt)
        J = jakobian(wsp_x, wsp_y, dN_dksi, dN_deta)
        detJ = det_jakobianu(J)
        if detJ == 0:
            continue
        J_inv = odwrotnosc_jakobianu(J)
        dN_dx = J_inv[0, 0] * dN_dksi + J_inv[0, 1] * dN_deta
        dN_dy = J_inv[1, 0] * dN_dksi + J_inv[1, 1] * dN_deta
        H_local = k * (np.outer(dN_dx, dN_dx) + np.outer(dN_dy, dN_dy)) * detJ * waga
        H += H_local
    return H


# Funkcje kształtu 1D na boku (parametr ksi w [-1,1], eta stały lub odwrotnie)
def shape_func_1D(ksi):
    # dla odcinka 2-węzłowego
    N1 = 0.5*(1 - ksi)
    N2 = 0.5*(1 + ksi)
    return np.array([N1, N2])


# Obliczanie macierzy Hbc i wektora P na krawędziach elementu
def oblicz_Hbc_i_P_lokalne(wsp_x, wsp_y, BC_nodes, alfa, Tot):
    # Element 4-węzłowy ma 4 boki:
    # Bok 1: węzły [0,1]
    # Bok 2: węzły [1,2]
    # Bok 3: węzły [2,3]
    # Bok 4: węzły [3,0]

    # Jeśli na danym boku węzły mają BC=1, to liczymy Hbc i P
    # Punkty całkowania 1D:
    pc_1D = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
    w_1D = np.array([1, 1])

    Hbc = np.zeros((4, 4))
    P = np.zeros(4)

    # Lista boków i odpowiadające węzły
    sides = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0)
    ]

    for side_id, (n1, n2) in enumerate(sides):
        # Sprawdzamy czy oba węzły na boku mają BC=1
        if BC_nodes[n1] == 1 and BC_nodes[n2] == 1:
            # Współrzędne węzłów tego boku:
            x1, y1 = wsp_x[n1], wsp_y[n1]
            x2, y2 = wsp_x[n2], wsp_y[n2]

            for i_pc, ksi in enumerate(pc_1D):
                w = w_1D[i_pc]
                N_1D = shape_func_1D(ksi)  # [N1, N2] na boku

                # Obliczanie długości boku
                dx = (x2 - x1)
                dy = (y2 - y1)
                dl = np.sqrt(dx*dx + dy*dy)

                # Jacobian dla 1D
                detJ_1D = dl / 2

                # Funkcje kształtu 2D na boku w zależności od boku
                N2D = np.zeros(4)
                if side_id == 0:  # bok 1: [0,1] eta=-1
                    N2D[0] = N_1D[0]  # N1
                    N2D[1] = N_1D[1]  # N2
                elif side_id == 1:  # bok 2: [1,2] ksi=1
                    N2D[1] = N_1D[0]  # N2
                    N2D[2] = N_1D[1]  # N3
                elif side_id == 2:  # bok 3: [2,3] eta=1
                    N2D[2] = N_1D[0]  # N3
                    N2D[3] = N_1D[1]  # N4
                elif side_id == 3:  # bok 4: [3,0] ksi=-1
                    N2D[3] = N_1D[0]  # N4
                    N2D[0] = N_1D[1]  # N1

                # Obliczamy wkład do Hbc: alfa * N2D^T * N2D * detJ_1D * w
                # Obliczamy wkład do P: - alfa * Tot * N2D * detJ_1D * w
                Hbc_local = alfa * np.outer(N2D, N2D) * detJ_1D * w
                P_local = alfa * Tot * N2D * detJ_1D * w

                Hbc += Hbc_local
                P += P_local

    return Hbc, P
