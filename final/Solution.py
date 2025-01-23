import numpy as np

class Solution:
    def __init__(self, elements, data):
        self.elements = elements
        self.data = data
        self.matrix_h_global = np.zeros((data.NodesNumber,data.NodesNumber))
        self.matrix_c_global = np.zeros((data.NodesNumber,data.NodesNumber))
        self.vector_p_global = np.zeros(data.NodesNumber)
        
        self.setup()
    
    def setup(self):
        self.agregate()

    # Skladanie lokalnych h, hbc, c i wektora p do struktury globalnej
    def agregate(self):
        for element in self.elements:
            for i in range(len(element.node_ids)):
                global_i = element.node_ids[i] - 1
                for j in range(len(element.node_ids)):
                    global_j = element.node_ids[j] - 1

                    self.matrix_h_global[global_i, global_j] += element.H[i][j]
                    self.matrix_h_global[global_i, global_j] += element.Hbc[i][j]
                    self.matrix_c_global[global_i, global_j] += element.C[i][j]
                self.vector_p_global[global_i] += element.P[i]

    # Implementacja petli czasowej
    # Do liczenia macierzy c w kolejnych krokach czasowych
    def solve(self):
        dt = self.data.SimulationStepTime
        matrix_c_dt = self.matrix_c_global / dt
        matrix_global = self.matrix_h_global + matrix_c_dt

        t0 = np.full(self.data.NodesNumber, self.data.InitialTemp)

        t = t0.copy()
        for i in range(dt, self.data.SimulationTime + dt, dt):
            right_side = np.dot(matrix_c_dt, t0) + self.vector_p_global
            t = np.linalg.solve(matrix_global, right_side)
            print(f"In {i}s:")
            print("temp min:", min(t), "temp max:", max(t))
            t0 = t.copy()
