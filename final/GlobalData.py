class GlobalData:
    def __init__(self):
        initial_values = {
            'SimulationTime': 0,
            'SimulationStepTime': 0,
            'Conductivity': 0,
            'Alfa': 0,
            'Tot': 0,
            'InitialTemp': 0,
            'Density': 0,
            'SpecificHeat': 0,
            'NodesNumber': 0,
            'ElementsNumber': 0,
        }
        
        for attr, value in initial_values.items():
            setattr(self, attr, value)

def load_data(filename):
    global_data = GlobalData()
    with open(filename, 'r') as file:
        for line in file:
            words = line.split()
            key = " ".join(words[:-1])
            value = words[-1]
            if key == 'SimulationTime':
                global_data.SimulationTime = int(value)
            elif key == 'SimulationStepTime':
                global_data.SimulationStepTime = int(value)
            elif key == 'Conductivity':
                global_data.Conductivity = int(value)
            elif key == 'Alfa':
                global_data.Alfa = int(value)
            elif key == 'Tot':
                global_data.Tot = int(value)
            elif key == 'InitialTemp':
                global_data.InitialTemp = int(value)
            elif key == 'Density':
                global_data.Density = int(value)
            elif key == 'SpecificHeat':
                global_data.SpecificHeat = int(value)
            elif key == 'Nodes number':
                global_data.NodesNumber = int(value)
            elif key == 'Elements number':
                global_data.ElementsNumber = int(value)
    return global_data