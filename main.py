import math


class Node:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y

    def __repr__(self):
        return f'Node({self.id}:, {self.x}, {self.y})'


class Element:
    def __init__(self, id, node1, node2, node3, node4):
        self.id = id
        self.node_ids = [node1, node2, node3, node4]

    def __repr__(self):
        return f'Element({self.id}) z węzłami: {self.node_ids}'


class Grid:
    def __init__(self, global_data):
        self.nNodes = global_data.nN
        self.nElements = global_data.nE
        self.nodes = []
        self.elements = []

    def read_from_file(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()

        node_start_index = lines.index("*Node\n") + 1
        for i in range(node_start_index, node_start_index + self.nNodes):
            node_data = lines[i].split(',')
            node_id = int(node_data[0].strip())
            x = float(node_data[1].strip())
            y = float(node_data[2].strip())
            self.nodes.append(Node(node_id, x, y))

        element_start_index = lines.index("*Element, type=DC2D4\n") + 1
        for i in range(element_start_index, element_start_index + self.nElements):
            element_data = lines[i].split(',')
            element_id = int(element_data[0].strip())
            node1 = int(element_data[1].strip())
            node2 = int(element_data[2].strip())
            node3 = int(element_data[3].strip())
            node4 = int(element_data[4].strip())
            self.elements.append(Element(element_id, node1, node2, node3, node4))

    def __repr__(self):
        return f'Grid z {self.nNodes} węzłami i {self.nElements} elementami'


class GlobalData:
    def __init__(self):
        self.simulation_time = None
        self.simulation_step_time = None
        self.conductivity = None
        self.alfa = None
        self.tot = None
        self.initial_temp = None
        self.density = None
        self.specific_heat = None
        self.nN = None
        self.nE = None
        self.H = None
        self.W = None

    def read_from_file(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        self.simulation_time = float(lines[0].split()[1])
        self.simulation_step_time = float(lines[1].split()[1])
        self.conductivity = float(lines[2].split()[1])
        self.alfa = float(lines[3].split()[1])
        self.tot = float(lines[4].split()[1])
        self.initial_temp = float(lines[5].split()[1])
        self.density = float(lines[6].split()[1])
        self.specific_heat = float(lines[7].split()[1])
        self.nN = int(lines[8].split()[2])
        self.nE = int(lines[9].split()[2])
        self.H = math.sqrt(self.nE)
        self.W = math.sqrt(self.nE)

    def __repr__(self):
        return (
            f"SimulationTime: {self.simulation_time}\n"
            f"SimulationStepTime: {self.simulation_step_time}\n"
            f"Conductivity: {self.conductivity}\n"
            f"Alfa: {self.alfa}\n"
            f"Tot: {self.tot}\n"
            f"InitialTemp: {self.initial_temp}\n"
            f"Density: {self.density}\n"
            f"SpecificHeat: {self.specific_heat}\n"
            f"Liczba węzłów (nN): {self.nN}\n"
            f"Liczba elementów (nE): {self.nE}\n"
            f"Wysokość siatki (H): {self.H}\n"
            f"Szerokość siatki (W): {self.W}"
        )


global_data = GlobalData()
global_data.read_from_file('test.txt')
print(global_data)

grid = Grid(global_data)
grid.read_from_file('test.txt')
print(grid)

for node in grid.nodes:
    print(node)

for element in grid.elements:
    print(element)

