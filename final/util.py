def setFilename(test):
    if(test == 1):
        return "Test1_4_4.txt"
    elif(test == 2):
        return "Test2_4_4_MixGrid.txt"
    elif (test == 3):
        return "Test3_31_31_kwadrat.txt"
    
def print_matrix(matrix, label):
    print(label)
    for row in matrix:
        print(" ".join(f"{val:8.3f}" if val != 0 else f"{0:8.0f}" for val in row))
    print()
