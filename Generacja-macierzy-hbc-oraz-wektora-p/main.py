import math
import numpy as np
from h_local import oblicz_macierz_H_lokalna

class Node:
    def __init__(self, x, y, bc=0):
        self.x = x
        self.y = y
        self.bc = bc  # 1 - węzeł z warunkiem brzegowym, 0 - bez

class Element:
    def __init__(self, ids):
        self.id = ids  # lista 4 węzłów
        # Macierze H, Hbc, C i wektor P - inicjalizacja na zero
        self.H = [[0.0 for _ in range(4)] for _ in range(4)]
        self.Hbc = [[0.0 for _ in range(4)] for _ in range(4)]
        self.C = [[0.0 for _ in range(4)] for _ in range(4)]
        self.P = [0.0 for _ in range(4)]

# Funkcje z h_local.py:


# Funkcja do wczytywania danych wejściowych
def read_input_file(filename):
    with open(filename, 'r') as f:
        lines = f.read().strip().split('\n')

    sim_time = float(lines[0].split()[1])
    sim_step = float(lines[1].split()[1])
    conductivity = float(lines[2].split()[1])
    alfa = float(lines[3].split()[1])
    tot = float(lines[4].split()[1])
    initial_temp = float(lines[5].split()[1])
    density = float(lines[6].split()[1])
    specific_heat = float(lines[7].split()[1])

    node_number = int(lines[8].split()[2])
    elem_number = int(lines[9].split()[2])

    # Szukamy sekcji *Node
    node_start_index = 0
    for i, l in enumerate(lines):
        if l.strip().startswith("*Node"):
            node_start_index = i + 1
            break

    nodes = []
    for i in range(node_number):
        data = lines[node_start_index + i].split(',')
        data = [d.strip() for d in data]
        x = float(data[1])
        y = float(data[2])
        nodes.append(Node(x, y, 0))

    # Szukamy sekcji *Element
    elem_start_index = 0
    for i, l in enumerate(lines):
        if l.strip().startswith("*Element"):
            elem_start_index = i + 1
            break

    elements = []
    for i in range(elem_number):
        data = lines[elem_start_index + i].split(',')
        data = [d.strip() for d in data]
        ids = [int(x) for x in data[1:]]
        elements.append(Element(ids))

    # Sekcja *BC
    bc_index = 0
    for i, l in enumerate(lines):
        if l.strip().startswith("*BC"):
            bc_index = i + 1
            break

    bc_nodes = lines[bc_index].split(',')
    bc_nodes = [int(x.strip()) for x in bc_nodes if x.strip() != '']
    for bn in bc_nodes:
        nodes[bn - 1].bc = 1

    return (sim_time, sim_step, conductivity, alfa, tot, initial_temp, density, specific_heat, nodes, elements)


def compute_H_for_element(element, nodes, conductivity, alfa, tot):
    # Definiowanie punktów Gaussa i wag dla całkowania 2D
    punkty_1D = [-1 / math.sqrt(3), 1 / math.sqrt(3)]
    wagi_1D = [1, 1]
    punkty_calek = [(ksi, eta) for eta in punkty_1D for ksi in punkty_1D]
    wagi_calek = [w1 * w2 for w1 in wagi_1D for w2 in wagi_1D]

    # Zamiana kolejności punktów (jeśli wymagana, wg kodu w h_local.py)
    punkty_calek[2], punkty_calek[3] = punkty_calek[3], punkty_calek[2]

    # Pozyskanie współrzędnych węzłów elementu
    x_coords = [nodes[id - 1].x for id in element.id]
    y_coords = [nodes[id - 1].y for id in element.id]
    wsp_x = np.array(x_coords)
    wsp_y = np.array(y_coords)

    # Obliczenie macierzy H lokalnej z użyciem funkcji z h_local.py
    H_np = oblicz_macierz_H_lokalna(wsp_x, wsp_y, punkty_calek, wagi_calek, conductivity)

    # Przepisanie wyników do element.H (list pythonowskich)
    for a in range(4):
        for b in range(4):
            element.H[a][b] = H_np[a][b]

    # Obliczanie Hbc i P (brzegi)
    side_gauss_pc = [-1.0 / math.sqrt(3), 1.0 / math.sqrt(3)]
    side_gauss_w = [1.0, 1.0]
    node_ids = element.id
    sides = [
        (node_ids[0], node_ids[1]),
        (node_ids[1], node_ids[2]),
        (node_ids[2], node_ids[3]),
        (node_ids[3], node_ids[0])
    ]
    for side, (id1, id2) in enumerate(sides):
        n1 = nodes[id1 - 1]
        n2 = nodes[id2 - 1]
        if n1.bc == 1 and n2.bc == 1:
            dx = n2.x - n1.x
            dy = n2.y - n1.y
            length = math.sqrt(dx * dx + dy * dy)
            for ip, ksi in enumerate(side_gauss_pc):
                w = side_gauss_w[ip]
                N1_1D = 0.5 * (1 - ksi)
                N2_1D = 0.5 * (1 + ksi)
                shape_on_side = [0.0, 0.0, 0.0, 0.0]
                if side == 0:
                    shape_on_side[0], shape_on_side[1] = N1_1D, N2_1D
                elif side == 1:
                    shape_on_side[1], shape_on_side[2] = N1_1D, N2_1D
                elif side == 2:
                    shape_on_side[2], shape_on_side[3] = N1_1D, N2_1D
                elif side == 3:
                    shape_on_side[3], shape_on_side[0] = N1_1D, N2_1D
                for a in range(4):
                    for b in range(4):
                        element.Hbc[a][b] += alfa * shape_on_side[a] * shape_on_side[b] * w * (length / 2.0)
                    element.P[a] += alfa * tot * shape_on_side[a] * w * (length / 2.0)


def main():
    filename = "Test2_4_4_MixGrid.txt"
    sim_time, sim_step, conductivity, alfa, tot, init_temp, density, specific_heat, nodes, elements = read_input_file(filename)

    print("Obliczenia wykonane na siatce o nazwie -", filename)
    print("Poprawnie otworzono plik.")
    print(sim_time)
    print(sim_step)
    print(conductivity)
    print(alfa)
    print(tot)
    print(init_temp)
    print(density)
    print(specific_heat)
    print(len(nodes), len(elements))

    for i, n in enumerate(nodes, start=1):
        print("x = {}, Y = {},  status = {}".format(n.x, n.y, n.bc))

    for e in elements:
        print("ID{}".format(e.id))

    # Obliczamy H, Hbc, P dla każdego elementu
    for i, e in enumerate(elements, start=1):
        compute_H_for_element(e, nodes, conductivity, alfa, tot)
        print("H dla elementu -", i)
        for row in e.H:
            print(" ".join(map(str, row)))
        print("BC", " ".join(map(str, e.P)))


if __name__ == "__main__":
    main()
