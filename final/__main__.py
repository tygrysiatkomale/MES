from math import sqrt
import numpy as np
import UniversalElement as UE
import Element, Node, GlobalData, util, Solution

np.set_printoptions(linewidth=200)
np.set_printoptions(precision=4)


def main(test: int, npc: int, npcbc: int, debug: bool) -> None:
    """
    Parameters:
    - test (int): Identifier for the test case.
    - npc (int): Number of integral points.
    - npcbc (int): Number of integral points for boundary condition.
    - debug (bool): Flag to enable or disable debug mode.
    """
    filename = util.setFilename(test)
    
    if debug:
        print(f"Starting test = {filename}, with npc = {npc}, npcbc = {npcbc}")

    data = GlobalData.load_data(filename)
    
    if debug:
        print("SimulationTime:", data.SimulationTime)
        print("SimulationStepTime:", data.SimulationStepTime)
        print("Conductivity:", data.Conductivity)
        print("Alfa:", data.Alfa)
        print("Tot:", data.Tot)
        print("InitialTemp:", data.InitialTemp)
        print("Density:", data.Density)
        print("SpecificHeat:", data.SpecificHeat)
        print("Nodes number:", data.NodesNumber)
        print("Elements number:", data.ElementsNumber)

    nodes = Node.load_nodes(filename)
    Node.load_bc(filename, nodes)
    
    if debug:
        for id, node in enumerate(nodes):
            print(f"Node: {id} x = {node.x}, y = {node.y}, bc = {node.bc}")
    
    universalElement = UE.UniversalElement(npc, npcbc)

    if debug:
        print("dNdKsi:")
        for row in universalElement.dNdKsi:
            print(row)
        print("dNdEta:")
        for row in universalElement.dNdEta:
            print(row)
        print("N:")
        for row in universalElement.N:
            print(row)
        print("N_bc:")
        for row in universalElement.N_bc:
            print(row)

    elements = Element.load_elements(filename, npc, npcbc, nodes, universalElement, data)
    
    if debug:
        for id, element in enumerate(elements, start = 1):
            print(f"Element {id}: ")
            print("Node IDs: ", element.node_ids)
            print("Jacobians:")
            for jacobian in element.jacobian:
                print(jacobian)
            print("detJs: ", element.jacobian_det)
            print("Inverse jacobians:")
            for jacobian in element.invJacobian:
                print(jacobian)
            print("Matrix X: ")
            for row in element.matrix_x:
                print(row)
            print("Matrix Y: ")
            for row in element.matrix_y:
                print(row)
            print("Matrix H: ")
            for row in element.H:
                print(row)
            print("Matrix Hbc: ")
            for row in element.Hbc:
                print(row)
            print("Vector P: ", element.P)
            print("Matrix C: ")
            for row in element.C:
                print(row)

    solution = Solution.Solution(elements, data)
    
    if debug:
        util.print_matrix(solution.matrix_h_global, "Final Matrix H Global:")
        util.print_matrix(solution.matrix_c_global, "Final Matrix C Global:")
        np.set_printoptions(precision = 2)
        print("Final Vector P: ")
        print(solution.vector_p_global)

    solution.solve()


if __name__ == "__main__":
    main(1, 4, 4, True)
    

