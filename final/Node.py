class Node:
    def __init__(self, x, y, bc = False):
        self.x = x
        self.y = y
        self.bc = bc

def load_nodes(filename):
    nodes = []
    
    with open(filename, 'r') as file:
        node_section = False
        for line in file:
            if line.strip() == '*Node':
                node_section = True
                continue
            if node_section:
                words = line.split(',')
                if len(words) == 3:
                    x = float(words[1].strip())
                    y = float(words[2].strip())
                    node = Node(x, y)
                    nodes.append(node)
                else:
                    node_section = False
    return nodes

def load_bc(filename, nodes):
    with open(filename, 'r') as file:
        is_bc_section = False
        for line in file:
            line = line.strip()
            if line.startswith('*BC'):
                is_bc_section = True
                continue
            elif is_bc_section and line:
                bc = [int(part.strip()) for part in line.split(',')]
                for value in bc:
                    nodes[value-1].bc = True 
            elif not line:
                is_bc_section = False