import numpy as np

class Element:
    def __init__(self, node_ids, nodes, npc, npcbc, ue, data):
        self.node_ids = node_ids
        self.nodes = nodes
        self.npc = npc
        self.npcbc = npcbc
        self.ue = ue
        self.data = data
        
        self.setup()
        
    def setup(self):
        self.calculate_jacobian()
        self.calculate_jacobian_det()
        self.calculate_det_j_bc()
        self.calculate_matrix_x()
        self.calculate_matrix_y()
        self.calculate_matrix_h()
        self.calculate_matrix_Hbc()
        self.calculate_vector_P()
        self.calculate_matrix_C()

    # Liczenie jakobianu poprzez sumowanie iloczynow pochodnych funkcji ksztaltu
    # Jakobian sluzy do transformacji wspolrzednych lokalnych na globalne (rzeczywiste)
    def calculate_jacobian(self):
        self.jacobian = [
            np.array([
                [sum(self.ue.dNdKsi[pc][j] * self.nodes[self.node_ids[j] - 1].x for j in range(4)),
                 sum(self.ue.dNdKsi[pc][j] * self.nodes[self.node_ids[j] - 1].y for j in range(4))],
                [sum(self.ue.dNdEta[pc][j] * self.nodes[self.node_ids[j] - 1].x for j in range(4)),
                 sum(self.ue.dNdEta[pc][j] * self.nodes[self.node_ids[j] - 1].y for j in range(4))]
            ])
            for pc in range(self.npc * self.npc)
        ]

    # Obliczanie wyznacznika jakobianu i jego odwrotnosci
    def calculate_jacobian_det(self):
        self.jacobian_det = [np.linalg.det(j) for j in self.jacobian]
        self.invJacobian = [np.linalg.inv(j) for j in self.jacobian]

    # transformacja pochodnych funkcji ksztaltu do ukladu globalnego
    def calculate_matrix_x(self):
        self.matrix_x = [
            [
                self.invJacobian[pc][0, 0] * self.ue.dNdKsi[pc][j] + self.invJacobian[pc][0, 1] * self.ue.dNdEta[pc][j]
                for j in range(4)
            ]
            for pc in range(self.npc * self.npc)
        ]

    # transformacja pochodnych funkcji ksztaltu do ukladu globalnego
    def calculate_matrix_y(self):
        self.matrix_y = [
            [
                self.invJacobian[pc][1, 0] * self.ue.dNdKsi[pc][j] + self.invJacobian[pc][1, 1] * self.ue.dNdEta[pc][j]
                for j in range(4)
            ]
            for pc in range(self.npc * self.npc)
        ]

    # lokalna macierz sztywnosci h
    def calculate_matrix_h(self):
        self.H = np.zeros((4, 4))
        for k in range(self.npc):
            for l in range(self.npc):
                for i in range(4):
                    for j in range(4):
                        self.H[i][j] += ( 
                            (self.matrix_x[k * self.npc + l][i] * self.matrix_x[k * self.npc + l][j]
                            + self.matrix_y[k * self.npc + l][i] * self.matrix_y[k * self.npc + l][j])
                            * self.jacobian_det[k * self.npc + l] * self.data.Conductivity
                            * self.ue.ksi_eta_weights[2][k] * self.ue.ksi_eta_weights[2][l]
                        )

    # obliczanie dlugosci krawedzi elementu
    def calculate_det_j_bc(self):
        self.det_j_bc = []
        for i in range(4):
            dx = self.nodes[self.node_ids[i] - 1].x - self.nodes[self.node_ids[(i + 1) % 4] - 1].x
            dy = self.nodes[self.node_ids[i] - 1].y - self.nodes[self.node_ids[(i + 1) % 4] - 1].y
            self.det_j_bc.append(np.sqrt(dx * dx + dy * dy) / 2)

    # Liczenie lokalnej macierzy konwekcji Hbc
    # Przechodzenie przez kazda z czterech krawedzi elementu i sprawdzanie czy jest warunek brzegowy
    # Iteracja po funkcjach ksztaltu
    def calculate_matrix_Hbc(self):
        self.Hbc = np.zeros((4, 4))
        for edge in range(4):
            if(self.nodes[self.node_ids[edge] - 1].bc and self.nodes[self.node_ids[(edge + 1) % 4] - 1].bc):
                for point in range(self.npcbc):
                    for i in range(4):
                        for j in range(4):
                            self.Hbc[i][j] += (np.outer(self.ue.N_bc[edge * self.npcbc + point], self.ue.N_bc[edge * self.npcbc + point])[i,j]
                            * self.ue.ksi_eta_weights_bc[2][point] * self.data.Alfa * self.det_j_bc[edge])
        
    def calculate_vector_P(self):
        self.P = np.zeros(4)
        
        for edge in range(4):
            if(self.nodes[self.node_ids[edge] - 1].bc and self.nodes[self.node_ids[(edge + 1) % 4] - 1].bc):
                for point in range(self.npcbc):
                    for i in range(4):
                        self.P[i] += (self.ue.N_bc[edge * self.npcbc + point][i] * self.ue.ksi_eta_weights_bc[2][point] 
                        * self.data.Alfa * self.det_j_bc[edge] * self.data.Tot)
    
    def calculate_matrix_C(self):
        self.C = np.zeros((4,4))
        for i in range(self.npc):
            for j in range(self.npc):
                for k in range(4):
                    for l in range(4):
                        self.C[k][l] += (self.data.Density * self.data.SpecificHeat * self.jacobian_det[i * self.npc + j]
                                         * np.outer(self.ue.N[i * self.npcbc + j], self.ue.N[i * self.npcbc + j])[k,l]
                                         * self.ue.ksi_eta_weights[2][i] * self.ue.ksi_eta_weights[2][j])
    
def load_elements(filename, npc, npcbc, nodes, ue, data):
    elements = []
    element_section = False
    
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('*Element'):
                element_section = True
                continue
            elif element_section and line.startswith('*BC'):
                break
            elif element_section and line:
                words = line.split(',')
                node_ids = [int(word.strip()) for word in words[1:]]
                element = Element(node_ids, nodes, npc, npcbc, ue, data)
                elements.append(element)
            elif not line:
                element_section = False
    return elements