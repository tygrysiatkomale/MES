import numpy as np

# Konfiguracja wyświetlania macierzy
np.set_printoptions(
    linewidth=150,    # Maksymalna szerokość w wierszu
    precision=4,      # Liczba miejsc po przecinku
    suppress=True     # Ukrywanie notacji naukowej dla małych liczb
)

def czytaj_dane_z_pliku(plik):
    """Funkcja odczytująca dane z pliku."""
    with open(plik, 'r') as f:
        linie = f.readlines()

    dane_symulacji = {}
    nodes = []
    elements = []
    bc_nodes = []

    sekcja = None
    for linia in linie:
        linia = linia.strip()

        if not linia:
            continue

        if linia.startswith("*Node"):
            sekcja = "node"
            continue
        elif linia.startswith("*Element"):
            sekcja = "element"
            continue
        elif linia.startswith("*BC"):
            sekcja = "bc"
            continue

        if sekcja is None and "," not in linia:
            klucz, wartosc = linia.rsplit(maxsplit=1)
            dane_symulacji[klucz] = float(wartosc)

        elif sekcja == "node":
            dane = linia.split(",")
            node_id = int(dane[0])
            x, y = map(float, dane[1:])
            nodes.append((node_id, x, y))

        elif sekcja == "element":
            dane = linia.split(",")
            element_id = int(dane[0])
            node_ids = list(map(int, dane[1:]))
            elements.append((element_id, node_ids))

        elif sekcja == "bc":
            bc_nodes.extend(map(int, linia.split(",")))

    nodes = np.array(nodes, dtype=[('id', int), ('x', float), ('y', float)])
    elements = np.array(elements, dtype=[('id', int), ('nodes', int, 4)])
    bc_nodes = np.array(bc_nodes, dtype=int)

    return dane_symulacji, nodes, elements, bc_nodes


def oblicz_macierz_H(nodes, element, k_func, num_points=2):
    """Oblicza lokalną macierz H dla zadanego elementu."""
    node_ids = element['nodes']
    wsp_x = []
    wsp_y = []

    for node_id in node_ids:
        node = nodes[nodes['id'] == node_id]
        wsp_x.append(node['x'][0])
        wsp_y.append(node['y'][0])
    wsp_x = np.array(wsp_x)
    wsp_y = np.array(wsp_y)

    if num_points == 2:
        w = [1, 1]
        pc = [-1 / np.sqrt(3), 1 / np.sqrt(3)]
    else:
        raise ValueError("Liczba punktów całkowania musi być 2.")

    H = np.zeros((4, 4))

    for i in range(num_points):
        for j in range(num_points):
            ksi, eta = pc[i], pc[j]
            dN_dksi = np.array([
                -0.25 * (1 - eta), 0.25 * (1 - eta),
                0.25 * (1 + eta), -0.25 * (1 + eta)
            ])
            dN_deta = np.array([
                -0.25 * (1 - ksi), -0.25 * (1 + ksi),
                0.25 * (1 + ksi), 0.25 * (1 - ksi)
            ])
            J11 = np.dot(dN_dksi, wsp_x)
            J12 = np.dot(dN_dksi, wsp_y)
            J21 = np.dot(dN_deta, wsp_x)
            J22 = np.dot(dN_deta, wsp_y)

            J = np.array([[J11, J12], [J21, J22]])
            detJ = np.linalg.det(J)

            if detJ <= 0:
                raise ValueError(f"Nieprawidłowy wyznacznik Jacobianu: {detJ} dla elementu {element['id']}.")

            invJ = np.linalg.inv(J)
            dN_dx_dy = invJ @ np.vstack((dN_dksi, dN_deta))
            B = np.zeros((2, 4))
            B[0, :] = dN_dx_dy[0, :]
            B[1, :] = dN_dx_dy[1, :]

            k = k_func(ksi, eta)
            H_pc = k * (B.T @ B) * detJ
            H += H_pc * w[i] * w[j]

    return H


def agregacja_macierzy(global_H, local_H, element):
    """Agreguje lokalną macierz H do globalnej."""
    node_ids = element['nodes']
    for i, global_i in enumerate(node_ids):
        for j, global_j in enumerate(node_ids):
            global_H[global_i - 1, global_j - 1] += local_H[i, j]


def main():
    dane_symulacji, nodes, elements, bc_nodes = czytaj_dane_z_pliku('Test1_4_4.txt')
    num_nodes = len(nodes)
    global_H = np.zeros((num_nodes, num_nodes))

    for element in elements:
        local_H = oblicz_macierz_H(nodes, element, k_func, num_points=2)
        agregacja_macierzy(global_H, local_H, element)

    print("Macierz globalna H:")
    print(global_H)


def k_func(ksi, eta):
    return 25


main()
