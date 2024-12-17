import numpy as np
from h_local import czytajDaneZPliku, oblicz_macierz_H_lokalna, oblicz_Hbc_i_P_lokalne

if __name__ == "__main__":
    plik = "Test2_4_4_MixGrid.txt"
    print(f"Obliczenia wykonane na siatce o nazwie - {plik}")
    data = czytajDaneZPliku(plik)
    print("Poprawnie otworzono plik.")

    # Wypis parametrów:
    print(data["SimulationTime"])
    print(data["SimulationStepTime"])
    print(data["Conductivity"])
    print(data["Alfa"])
    print(data["Tot"])
    print(data["InitialTemp"])
    print(data["Density"])
    print(data["SpecificHeat"])

    # Dane siatki
    nn = data["NodesNumber"]
    ne = data["ElementsNumber"]

    print(f"{int(np.sqrt(nn))} {nn}")
    print(f"{ne} {ne}")

    nodes = data["Nodes"]
    elements = data["Elements"]
    bc_nodes_list = data["BC"]

    # Tworzymy tablice współrzędnych i tablicę BC:
    node_x = np.zeros(nn)
    node_y = np.zeros(nn)
    BC = np.zeros(nn, dtype=int)

    for (n_id, x, y) in nodes:
        node_x[n_id - 1] = x
        node_y[n_id - 1] = y

    for bcn in bc_nodes_list:
        BC[bcn - 1] = 1

    # Wypis węzłów
    for i in range(nn):
        print(f"x = {node_x[i]}   Y = {node_y[i]},  status = {BC[i]}")

    # Wypis elementów
    for e_id, e_nodes in elements:
        print(f"ID{e_nodes}")

    # Definiowanie punktów Gaussa w 2D
    punkty_1D = [-1 / np.sqrt(3), 1 / np.sqrt(3)]
    wagi_1D = [1, 1]
    punkty_calek_2D = [(ksi, eta) for eta in punkty_1D for ksi in punkty_1D]
    wagi_calek_2D = [w1 * w2 for w1 in wagi_1D for w2 in wagi_1D]

    k = data["Conductivity"]
    alfa = data["Alfa"]
    Tot = data["Tot"]

    # Inicjalizacja globalnych
    H_global = np.zeros((nn, nn))
    P_global = np.zeros(nn)

    # Obliczanie macierzy lokalnych i składanie do globalnych
    for (e_id, e_nodes) in elements:
        en = [n - 1 for n in e_nodes]

        wsp_x = node_x[en]
        wsp_y = node_y[en]

        # Macierz H lokalna
        H = oblicz_macierz_H_lokalna(wsp_x, wsp_y, punkty_calek_2D, wagi_calek_2D, k)

        # Macierz Hbc i wektor P lokalne
        Hbc, P = oblicz_Hbc_i_P_lokalne(wsp_x, wsp_y, BC[en], alfa, Tot)

        # Wyświetlanie macierzy elementów:
        print(f"\nH dla elementu - {e_id}:")
        print(H)
        print(f"Hbc dla elementu - {e_id}:")
        print(Hbc)
        print(f"P dla elementu - {e_id}:")
        print(P)

        # Łączna macierz H dla elementu:
        H_total = H + Hbc

        # Montowanie do globalnych
        for i in range(4):
            P_global[en[i]] += P[i]
            for j in range(4):
                H_global[en[i], en[j]] += H_total[i, j]

    print("\nTemperature results")
    print("Macierz H_global:")
    for i in range(nn):
        line = " ".join([f"{H_global[i, j]:.5f}" for j in range(nn)])
        print(line)

    print("Wektor P_global:")
    line = " ".join([f"{val:.2f}" for val in P_global])
    print(line)

    # Rozwiązywanie układu równań [H]{t} = {P}
    try:
        t = np.linalg.solve(H_global, P_global)
        print("\nWektor temperatur {t}:")
        for i in range(nn):
            print(f"Temperatura węzła {i+1}: {t[i]:.2f}")
    except np.linalg.LinAlgError as e:
        print("Błąd w rozwiązywaniu układu równań:", e)
