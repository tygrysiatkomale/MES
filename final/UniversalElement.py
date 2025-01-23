import numpy as np


class UniversalElement:
    def __init__(self, npc, npcbc):
        self.npc = npc
        self.npcbc = npcbc
        
        self.setup()
        
    def setup(self):
        self.set_ksi_eta_weights()
        self.calculate_dNs()
        self.calculate_N_bc()
        self.calculate_N()
        
    def set_ksi_eta_weights(self):
        self.ksi_eta_weights = self.initialize_ksi_eta()
        self.ksi_eta_weights_bc = self.initalize_ksi_eta_bc()
        
    def initialize_ksi_eta(self):
        ksi, eta, weights = [], [], []
        if self.npc == 2:
            ksi = np.array([-1 / np.sqrt(3), 1 / np.sqrt(3), -1 / np.sqrt(3), 1 / np.sqrt(3)])
            eta = np.array([-1 / np.sqrt(3), -1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)])
            weights = np.array([1, 1])
        elif self.npc == 3:
            ksi = np.array([-np.sqrt(3 / 5), 0, np.sqrt(3 / 5), -np.sqrt(3 / 5), 0, np.sqrt(3 / 5), -np.sqrt(3 / 5), 0, np.sqrt(3 / 5)])
            eta = np.array([-np.sqrt(3 / 5), -np.sqrt(3 / 5), -np.sqrt(3 / 5), 0, 0, 0, np.sqrt(3 / 5), np.sqrt(3 / 5), np.sqrt(3 / 5)])
            weights = np.array([5/9, 8/9, 5/9])
        elif self.npc == 4:
            vals = [-np.sqrt(3 / 7 + 2 / 7 * np.sqrt(6 / 5)),
                    -np.sqrt(3 / 7 - 2 / 7 * np.sqrt(6 / 5)),
                    np.sqrt(3 / 7 - 2 / 7 * np.sqrt(6 / 5)),
                    np.sqrt(3 / 7 + 2 / 7 * np.sqrt(6 / 5))]
            ksi = np.repeat(vals, 4) # 1,2,3,4,1,2,3,4
            eta = np.tile(vals, 4)   # 1,1,1,1,2,2,2,2
            weights = np.array([(18 - np.sqrt(30)) / 36, (18 + np.sqrt(30)) / 36, (18 + np.sqrt(30)) / 36, (18 - np.sqrt(30)) / 36])
        else:
            raise ValueError("Invalid number of nodes")
        return ksi, eta, weights

    def initalize_ksi_eta_bc(self):
        ksi, eta, weights = [], [], []
        if self.npcbc == 2:
            ksi = np.array([-1 / np.sqrt(3), 1 / np.sqrt(3), 1, 1, 1 / np.sqrt(3), -1 / np.sqrt(3), -1, -1])
            eta = np.array([-1, -1, -1 / np.sqrt(3), 1 / np.sqrt(3), 1, 1, 1 / np.sqrt(3), -1 / np.sqrt(3)])
            weights = np.array([1, 1])
        elif self.npcbc == 3:
            ksi = np.array(
                [-np.sqrt(3 / 5), 0, np.sqrt(3 / 5),
                 1, 1, 1,
                 np.sqrt(3 / 5), 0, -np.sqrt(3 / 5),
                 -1, -1, -1])
            eta = np.array(
                [-1, -1, -1,
                 -np.sqrt(3 / 5), 0, np.sqrt(3 / 5),
                 1, 1, 1,
                 np.sqrt(3 / 5), 0, -np.sqrt(3 / 5)])
            weights = np.array([5 / 9, 8 / 9, 5 / 9])
        elif self.npcbc == 4:
            ksi = np.array(
                [-np.sqrt(3 / 7 + 2 / 7 * np.sqrt(6 / 5)), -np.sqrt(3 / 7 - 2 / 7 * np.sqrt(6 / 5)), np.sqrt(3 / 7 - 2 / 7 * np.sqrt(6 / 5)), np.sqrt(3 / 7 + 2 / 7 * np.sqrt(6 / 5)),
                 1, 1, 1, 1,
                 np.sqrt(3 / 7 + 2 / 7 * np.sqrt(6 / 5)), np.sqrt(3 / 7 - 2 / 7 * np.sqrt(6 / 5)), -np.sqrt(3 / 7 - 2 / 7 * np.sqrt(6 / 5)), -np.sqrt(3 / 7 + 2 / 7 * np.sqrt(6 / 5)),
                 -1, -1, -1, -1])
            eta = np.array([
                -1, -1, -1, -1,
                -np.sqrt(3 / 7 + 2 / 7 * np.sqrt(6 / 5)), -np.sqrt(3 / 7 - 2 / 7 * np.sqrt(6 / 5)), np.sqrt(3 / 7 - 2 / 7 * np.sqrt(6 / 5)), np.sqrt(3 / 7 + 2 / 7 * np.sqrt(6 / 5)),
                1, 1, 1, 1,
                np.sqrt(3 / 7 + 2 / 7 * np.sqrt(6 / 5)), np.sqrt(3 / 7 - 2 / 7 * np.sqrt(6 / 5)), -np.sqrt(3 / 7 - 2 / 7 * np.sqrt(6 / 5)), -np.sqrt(3 / 7 + 2 / 7 * np.sqrt(6 / 5))])
            weights = np.array([(18 - np.sqrt(30)) / 36, (18 + np.sqrt(30)) / 36, (18 + np.sqrt(30)) / 36, (18 - np.sqrt(30)) / 36])
        else:
            raise ValueError("Invalid number of nodes")
        return ksi, eta, weights
    
    def calculate_dNs(self):
        self.dNdKsi = self.calculate_dNdKsi()
        self.dNdEta = self.calculate_dNdEta()
        
    def calculate_dNdKsi(self):
        dNdKsi = []
        for i in range(self.npc * self.npc):
            eta = self.ksi_eta_weights[1][i]
            dNdKsi.append([
                    -0.25 * (1 - eta),
                    0.25 * (1 - eta),
                    0.25 * (1 + eta),
                    -0.25 * (1 + eta)
                ])
        
        return dNdKsi

    def calculate_dNdEta(self):
        dNdEta = []
        for i in range(self.npc * self.npc):
            ksi = self.ksi_eta_weights[0][i]
            dNdEta.append([
                    -0.25 * (1 - ksi),
                    -0.25 * (1 + ksi),
                    0.25 * (1 + ksi),
                    0.25 * (1 - ksi)
                ])
            
        return dNdEta
    
    def calculate_N_bc(self):
        self.N_bc = []
        for i in range(self.npcbc*4):
            self.N_bc.append([
            1 / 4 * (1 - self.ksi_eta_weights_bc[0][i]) * (1 - self.ksi_eta_weights_bc[1][i]),
            1 / 4 * (1 + self.ksi_eta_weights_bc[0][i]) * (1 - self.ksi_eta_weights_bc[1][i]),
            1 / 4 * (1 + self.ksi_eta_weights_bc[0][i]) * (1 + self.ksi_eta_weights_bc[1][i]),
            1 / 4 * (1 - self.ksi_eta_weights_bc[0][i]) * (1 + self.ksi_eta_weights_bc[1][i])])


    # Wyznaczanie wartosci funkcji ksztaltu w punktach gaussa
    def calculate_N(self):
        self.N = []
        for i in range(self.npc * self.npc):
            self.N.append([
                1 / 4 * (1 - self.ksi_eta_weights[0][i]) * (1 - self.ksi_eta_weights[1][i]),
                1 / 4 * (1 + self.ksi_eta_weights[0][i]) * (1 - self.ksi_eta_weights[1][i]),
                1 / 4 * (1 + self.ksi_eta_weights[0][i]) * (1 + self.ksi_eta_weights[1][i]),
                1 / 4 * (1 - self.ksi_eta_weights[0][i]) * (1 + self.ksi_eta_weights[1][i])
            ])
